import copy
import gc
import glob
import itertools
import logging
import multiprocessing as mp
import os
import random
import shutil
import warnings
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from stdatamodels.jwst import datamodels
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..utils import get_dq_bit_mask, reproject_image, make_source_mask

matplotlib.use("agg")
log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def make_stacked_image(
    files,
    out_name,
):
    """Create a quick stacked image from a series of input images

    Args:
        files: List of input files
        out_name: Output stacked file
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        hdus = []

        for file in files:
            hdu = fits.open(file)

            dq_bit_mask = get_dq_bit_mask(hdu["DQ"].data)

            hdu["SCI"].data[dq_bit_mask != 0] = np.nan

            hdus.append(hdu)

        output_projection, shape_out = find_optimal_celestial_wcs(hdus, hdu_in="SCI")
        stacked_image, stacked_foot = reproject_and_coadd(
            hdus,
            output_projection=output_projection,
            shape_out=shape_out,
            hdu_in="SCI",
            reproject_function=reproject_interp,
        )

        hdr = output_projection.to_header()

        hdu = fits.ImageHDU(data=stacked_image, header=hdr, name="SCI")
        hdu.writeto(
            out_name,
            overwrite=True,
        )

        del hdus
        gc.collect()

    return True


class LevelMatchStep:
    def __init__(
        self,
        in_dir,
        out_dir,
        step_ext,
        procs,
        do_local_subtraction=True,
        sigma=3,
        npixels=3,
        dilate_size=7,
        max_iters=20,
        max_points=10000,
        do_sigma_clip=False,
        weight_method="equal",
        min_area_percent=0.002,
        min_linear_frac=0.25,
        rms_sig_limit=2,
        overwrite=False,
    ):
        """Perform background matching between tiles

        This step performs background matching by minimizing the
        per-pixel differences between overlapping tiles. It does
        this first for dither groups, before creating a stacked
        image of these (to maximize areal coverage) and minimizing
        between all stacked images within a mosaic. This is necessary
        for observations that don't really have a background, and
        performs significantly better than the JWST implementation.

        N.B. If you use this, skymatch in the level 3 pipeline stage
        should be global or off, to avoid undoing this work

        Args:
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of parallel processes to run
            do_local_subtraction: Whether to do a sigma-clipped local median
                subtraction. Defaults to True
            sigma: Sigma for sigma-clipping. Defaults to 3
            npixels: Pixels to grow for masking. Defaults to 5
            dilate_size: make_source_mask dilation size. Defaults to 7
            max_iters: Maximum sigma-clipping iterations. Defaults to 20
            max_points: Maximum points to include in histogram plots. This step can
                be slow so this speeds it up. Defaults to 10000
            do_sigma_clip: Whether to do sigma-clipping on data when reprojecting.
                Defaults to False
            weight_method: How to weight in least-squares minimization. Options are
                'equal' (equal weighting), 'npix' (weight by number of pixels), and
                'rms' (weight by rms of the delta values). Defaults to 'equal'
            min_area_percent: Minimum percentage of average areal overlap to remove tiles.
                Defaults to 0.002 (0.2%)
            min_linear_frac: Minimum linear overlap in any direction to keep tiles.
                Defaults to 0.25
            rms_sig_limit: Sigma limit for cutting off noisy fits. Defaults to 2
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs
        self.do_local_subtraction = do_local_subtraction
        self.sigma = sigma
        self.npixels = npixels
        self.dilate_size = dilate_size
        self.max_iters = max_iters
        self.max_points = max_points
        self.do_sigma_clip = do_sigma_clip
        self.weight_method = weight_method
        self.min_area_percent = min_area_percent
        self.min_linear_frac = min_linear_frac
        self.rms_sig_limit = rms_sig_limit
        self.overwrite = overwrite

        self.plot_dir = os.path.join(
            self.out_dir,
            "plots",
        )
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def do_step(self):
        """Run level matching"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            os.makedirs(self.plot_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "level_match_step_complete.txt",
        )
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        files = glob.glob(
            os.path.join(
                self.in_dir,
                f"*_{self.step_ext}.fits",
            )
        )
        files.sort()

        # Split these into dithers per-chip
        dithers = []
        for file in files:
            file_split = os.path.split(file)[-1].split("_")
            dithers.append("_".join(file_split[:2]) + "_*_" + file_split[-2])
        dithers = np.unique(dithers)
        dithers.sort()

        # First pass where we do this per-dither, per-chip. Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(dithers)])

        deltas, dither_files = self.get_per_dither_delta(
            dithers=dithers,
            procs=procs,
        )

        # Apply this calculated value
        for idx in range(len(deltas)):
            deltas_idx = copy.deepcopy(deltas[idx])
            dither_files_idx = copy.deepcopy(dither_files[idx])

            # If we're including a local subtraction, do it here
            if self.do_local_subtraction:
                with datamodels.open(dither_files_idx[0]) as im:
                    data = copy.deepcopy(im.data)

                    # Mask out bad data
                    dq_bit_mask = get_dq_bit_mask(im.dq)
                    data[dq_bit_mask != 0] = np.nan
                    data[data == 0] = np.nan

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        mask = make_source_mask(
                            data,
                            nsigma=self.sigma,
                            dilate_size=self.dilate_size,
                            sigclip_iters=self.max_iters,
                        )

                        # Calculate sigma-clipped median
                        local_delta = sigma_clipped_stats(
                            data,
                            mask=mask,
                            sigma=self.sigma,
                            maxiters=self.max_iters,
                        )[1]
                del im
            else:
                local_delta = 0

            for i, dither_file in enumerate(dither_files_idx):
                short_file = os.path.split(dither_file)[-1]
                out_file = os.path.join(
                    self.out_dir,
                    short_file,
                )
                delta = copy.deepcopy(deltas_idx[i])
                log.info(f"{short_file}, delta={delta + local_delta:.2f}")

                with datamodels.open(dither_file) as im:

                    zero_idx = np.where(im.data == 0)

                    im.data -= delta + local_delta
                    im.data[zero_idx] = 0
                    im.save(out_file)
                del im

        if len(dithers) > 1:
            # From the individually corrected images
            # get a stacked image
            stacked_dir = self.out_dir + "_stacked"
            if not os.path.exists(stacked_dir):
                os.makedirs(stacked_dir)

            successes = self.make_stacked_images(
                dithers=dithers,
                stacked_dir=stacked_dir,
                procs=procs,
            )

            if not np.all(successes):
                log.warning("Failures detected making stacked images")
                return False

            # Now match up these stacked images
            stacked_files = glob.glob(
                os.path.join(stacked_dir, f"*_{self.step_ext}.fits")
            )
            (
                delta_matrix,
                npix_matrix,
                rms_matrix,
                lin_size_matrix,
            ) = self.calculate_delta(
                stacked_files,
                procs=procs,
                stacked_image=True,
            )
            deltas = self.find_optimum_deltas(
                delta_mat=delta_matrix,
                npix_mat=npix_matrix,
                rms_mat=rms_matrix,
                lin_size_mat=lin_size_matrix,
            )

            # Subtract the per-dither delta, do in place
            for idx, delta in enumerate(deltas):
                short_stack_file = os.path.split(stacked_files[idx])[-1]
                dither_files = glob.glob(
                    os.path.join(
                        self.out_dir,
                        short_stack_file,
                    )
                )
                dither_files.sort()

                for dither_file in dither_files:
                    short_dither_file = os.path.split(dither_file)[-1]
                    log.info(f"{short_dither_file}, delta={delta:.2f}")

                    with datamodels.open(dither_file) as im:

                        zero_idx = np.where(im.data == 0)

                        im.data -= delta
                        im.data[zero_idx] = 0
                        im.save(dither_file)
                    del im

            # Remove the stacked images
            shutil.rmtree(stacked_dir)

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def get_per_dither_delta(
        self,
        dithers,
        procs=1,
    ):
        """Function to parallelise getting the delta for each observation in a dither sequence

        Args:
            dithers: List of dithers to get deltas for
            procs: Number of processes to run simultaneously. Defaults
                to 1
        """

        log.info("Getting deltas for individual dithers")

        deltas = []
        dither_files = []

        with mp.get_context("fork").Pool(procs) as pool:
            for delta, dither_file in tqdm(
                pool.imap_unordered(
                    partial(
                        self.parallel_per_dither_delta,
                    ),
                    dithers,
                ),
                total=len(dithers),
                desc="Matching individual dithers",
                ascii=True,
            ):
                deltas.append(delta)
                dither_files.append(dither_file)

            pool.close()
            pool.join()
            gc.collect()

        return deltas, dither_files

    def parallel_per_dither_delta(
        self,
        dither,
    ):
        """Function to parallelise up matching dithers

        Args:
            dither: Input dither
        """

        dither_files = glob.glob(
            os.path.join(
                self.in_dir,
                f"{dither}*_{self.step_ext}.fits",
            )
        )

        dither_files.sort()

        with threadpool_limits(limits=1, user_api=None):
            (
                delta_matrix,
                npix_matrix,
                rms_matrix,
                lin_size_matrix,
            ) = self.calculate_delta(dither_files)
            deltas = self.find_optimum_deltas(
                delta_mat=delta_matrix,
                npix_mat=npix_matrix,
                rms_mat=rms_matrix,
                lin_size_mat=lin_size_matrix,
            )

        return deltas, dither_files

    def make_stacked_images(
        self,
        dithers,
        stacked_dir,
        procs=1,
    ):
        """Function to parallellise up making stacked dither images

        Args:
            dithers: List of dithers to go
            stacked_dir: Where to save stacked images to
            procs: Number of simultaneous processes to run.
                Defaults to 1
        """

        log.info("Created stacked images")

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in tqdm(
                pool.imap_unordered(
                    partial(
                        self.parallel_make_stacked_image,
                        out_dir=stacked_dir,
                    ),
                    dithers,
                ),
                total=len(dithers),
                desc="Creating stacked images",
                ascii=True,
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_make_stacked_image(
        self,
        dither,
        out_dir,
    ):
        """Light wrapper around parallelising the stacked image

        Args:
            dither: Dither to stack
            out_dir: Directory to save to
        """

        files = glob.glob(
            os.path.join(
                self.out_dir,
                f"{dither}*_{self.step_ext}.fits",
            )
        )
        files.sort()

        # Create output name
        file_name_split = os.path.split(files[0])[-1].split("_")
        file_name_split[2] = "*"
        out_name = "_".join(file_name_split)
        out_name = os.path.join(out_dir, out_name)

        success = make_stacked_image(
            files=files,
            out_name=out_name,
        )
        if not success:
            return False

        return True

    def calculate_delta(
        self,
        files,
        stacked_image=False,
        procs=None,
    ):
        """Match relative offsets between tiles

        Args:
            files (list): List of files to match
            stacked_image: Whether this is a stacked image or not.
                Default to False
            procs (int, optional): Number of processes to run in
                parallel. Defaults to None, which is series
        """

        files = files

        deltas = np.zeros([len(files), len(files)])
        weights = np.zeros_like(deltas)
        rmses = np.zeros_like(deltas)
        lin_sizes = np.ones_like(deltas)

        # Reproject all the HDUs. Start by building the optimal WCS
        if isinstance(files[0], list):
            files_flat = list(itertools.chain(*files))
        else:
            files_flat = copy.deepcopy(files)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimal_wcs, optimal_shape = find_optimal_celestial_wcs(
                files_flat,
                hdu_in="SCI",
            )

        if procs is None:
            # Use a serial method

            # Reproject files, maintaining structure
            file_reproj = []

            for file in files:
                file_reproj.append(
                    self.get_reproject(
                        file=file,
                        optimal_wcs=optimal_wcs,
                        optimal_shape=optimal_shape,
                        stacked_image=stacked_image,
                    )
                )

            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    plot_name = self.get_plot_name(
                        files[i],
                        files[j],
                    )

                    n_pix, delta, rms, lin_size = self.get_level_match(
                        files1=file_reproj[i],
                        files2=file_reproj[j],
                        plot_name=plot_name,
                    )

                    # These are symmetrical by design
                    if n_pix == 0 or delta is None:
                        continue

                    deltas[j, i] = delta
                    weights[j, i] = n_pix
                    rmses[j, i] = rms
                    lin_sizes[j, i] = lin_size

                    deltas[i, j] = -delta
                    weights[i, j] = n_pix
                    rmses[i, j] = rms
                    lin_sizes[i, j] = lin_size

            gc.collect()

        else:
            # We can multiprocess this, since each calculation runs independently

            n_procs = np.nanmin([procs, len(files)])

            with mp.get_context("fork").Pool(n_procs) as pool:
                file_reproj = list([None] * len(files))

                for i, result in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_get_reproject,
                            files=files,
                            optimal_wcs=optimal_wcs,
                            optimal_shape=optimal_shape,
                            stacked_image=stacked_image,
                        ),
                        range(len(files)),
                    ),
                    total=len(files),
                    desc="Reprojecting files",
                    ascii=True,
                ):
                    file_reproj[i] = result

                pool.close()
                pool.join()
                gc.collect()

            all_ijs = [
                (i, j) for i in range(len(files)) for j in range(i + 1, len(files))
            ]

            ijs = []
            delta_vals = []
            n_pix_vals = []
            rms_vals = []
            lin_size_vals = []

            for ij in tqdm(all_ijs, ascii=True, desc="Calculating delta matrix"):
                ij, delta, n_pix, rms, lin_size = self.parallel_delta_matrix(
                    ij=ij,
                    files=files,
                    file_reproj=file_reproj,
                )

                ijs.append(ij)
                delta_vals.append(delta)
                n_pix_vals.append(n_pix)
                rms_vals.append(rms)
                lin_size_vals.append(lin_size)

            for idx, ij in enumerate(ijs):
                i = ij[0]
                j = ij[1]

                if n_pix_vals[idx] == 0 or delta_vals[idx] is None:
                    continue

                deltas[j, i] = delta_vals[idx]
                weights[j, i] = n_pix_vals[idx]
                rmses[j, i] = rms_vals[idx]
                lin_sizes[j, i] = lin_size_vals[idx]

                deltas[i, j] = -delta_vals[idx]
                weights[i, j] = n_pix_vals[idx]
                rmses[i, j] = rms_vals[idx]
                lin_sizes[i, j] = lin_size_vals[idx]

            gc.collect()

        return deltas, weights, rmses, lin_sizes

    def parallel_delta_matrix(
        self,
        ij,
        file_reproj,
        files,
    ):
        """Function to parallelise up getting delta matrix values

        Args:
            ij: List of matrix (i, j) values
            file_reproj: Reprojected file
            files: Full list of files
        """

        i = ij[0]
        j = ij[1]

        plot_name = None
        if self.plot_dir is not None:
            plot_name = self.get_plot_name(
                files1=files[i],
                files2=files[j],
            )

        with threadpool_limits(limits=1, user_api=None):
            n_pix, delta, rms, lin_size = self.get_level_match(
                files1=file_reproj[i],
                files2=file_reproj[j],
                plot_name=plot_name,
            )

        gc.collect()

        return ij, delta, n_pix, rms, lin_size

    def get_level_match(
        self,
        files1,
        files2,
        plot_name=None,
        maxiters=10,
    ):
        """Calculate relative difference between groups of files on the same pixel grid

        Args:
            files1: List of files to get difference from
            files2: List of files to get relative difference to
            plot_name: Output plot name. Defaults to None
            maxiters: Maximum iterations for the sigma-clipping. Defaults
                to 10
        """

        diffs = []

        if not isinstance(files1, list):
            files1 = [files1]
        if not isinstance(files2, list):
            files2 = [files2]

        n_pix = 0

        fig, axs = None, None
        if plot_name is not None:
            fig, axs = plt.subplots(
                nrows=len(files1),
                ncols=len(files2),
                figsize=(2.5 * len(files2), 2.5 * len(files1)),
                squeeze=False,
            )

        lin_size = 0

        for file_idx1, file1 in enumerate(files1):
            # Get out coordinates where data is valid, so we can do a linear
            # extent test
            ii, jj = np.indices(file1.array.shape, dtype=float)
            ii[np.where(np.isnan(file1.array))] = np.nan
            jj[np.where(np.isnan(file1.array))] = np.nan

            # If we have something that's all NaNs
            # (e.g. lyot on MIRI subarray obs.), skip
            if np.all(np.isnan(ii)):
                continue

            file1_iaxis = np.nanmax(ii) - np.nanmin(ii)
            file1_jaxis = np.nanmax(jj) - np.nanmin(jj)

            for file_idx2, file2 in enumerate(files2):
                if file2.overlaps(file1):
                    # Get diffs, remove NaNs
                    diff = file2 - file1
                    diff_arr = diff.array
                    diff_foot = diff.footprint
                    diff_arr[diff == 0] = np.nan
                    diff_arr[diff_foot == 0] = np.nan

                    # Get out coordinates where data is valid, so we can do a linear
                    # extent test
                    ii, jj = np.indices(file2.array.shape, dtype=float)
                    ii[np.where(np.isnan(file2.array))] = np.nan
                    jj[np.where(np.isnan(file2.array))] = np.nan

                    # If we have something that's all NaNs
                    # (e.g. lyot on MIRI subarray obs.), skip
                    if np.all(np.isnan(ii)):
                        continue

                    file2_iaxis = np.nanmax(ii) - np.nanmin(ii)
                    file2_jaxis = np.nanmax(jj) - np.nanmin(jj)

                    ii, jj = np.indices(diff_arr.shape, dtype=float)
                    ii[np.where(np.isnan(diff_arr))] = np.nan
                    jj[np.where(np.isnan(diff_arr))] = np.nan

                    # If the slices are all NaNs, then we can just move on
                    if np.all(np.isnan(ii)):
                        continue
                    diff_iaxis = np.nanmax(ii) - np.nanmin(ii)
                    diff_jaxis = np.nanmax(jj) - np.nanmin(jj)

                    # Include everything, but flag if we've hit the minimum linear extent
                    if (
                        diff_iaxis > file1_iaxis * self.min_linear_frac
                        or diff_jaxis > file1_jaxis * self.min_linear_frac
                        or diff_iaxis > file2_iaxis * self.min_linear_frac
                        or diff_jaxis > file2_jaxis * self.min_linear_frac
                    ):
                        lin_size = 1

                    diff = diff_arr[np.isfinite(diff_arr)].tolist()
                    n_pix += len(diff)

                    diffs.extend(diff)

                    if plot_name is not None:
                        if len(diff) > 0:
                            vmin, vmax = np.nanpercentile(diff, [1, 99])
                            axs[file_idx1, file_idx2].imshow(
                                diff_arr,
                                origin="lower",
                                vmin=vmin,
                                vmax=vmax,
                                interpolation="nearest",
                            )
                if plot_name is not None:
                    axs[file_idx1, file_idx2].set_axis_off()

        if plot_name is not None:
            if n_pix > 0:
                plt.savefig(f"{plot_name}_ims.png", bbox_inches="tight")
                plt.savefig(f"{plot_name}_ims.pdf", bbox_inches="tight")
            plt.close()

        if n_pix > 0:
            # Sigma-clip to remove outliers in the distribution
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, delta, rms = sigma_clipped_stats(diffs, maxiters=maxiters)

            if plot_name is not None:
                # Get histogram range
                diffs_hist = None

                if self.max_points is not None:
                    if len(diffs) > self.max_points:
                        diffs_hist = random.sample(diffs, self.max_points)
                if diffs_hist is None:
                    diffs_hist = copy.deepcopy(diffs)

                hist_range = np.nanpercentile(diffs_hist, [1, 99])

                plt.figure(figsize=(5, 4))
                plt.hist(
                    diffs_hist,
                    histtype="step",
                    bins=50,
                    range=hist_range,
                    color="gray",
                )
                plt.axvline(
                    delta,
                    c="k",
                    ls="--",
                )

                plt.xlabel("Diff (MJy/sr)")
                plt.ylabel("N")

                plt.tight_layout()

                plt.savefig(f"{plot_name}_hist.pdf", bbox_inches="tight")
                plt.savefig(f"{plot_name}_hist.png", bbox_inches="tight")
                plt.close()

        else:
            delta = None
            rms = 0

        gc.collect()

        return n_pix, delta, rms, lin_size

    def find_optimum_deltas(
        self,
        delta_mat,
        npix_mat,
        rms_mat,
        lin_size_mat,
    ):
        """Get optimum deltas from a delta/weight matrix.

        Taken from the JWST skymatch step, with some edits to remove potentially bad fits due
        to small areal overlaps, or noisy diffs, and various weighting schemes

        Args:
            delta_mat (np.ndarray): Matrix of delta values
            npix_mat (np.ndarray): Matrix of number of pixel values for calculating delta
            rms_mat (np.ndarray): Matrix of RMS values
            lin_size_mat (np.ndarray): 1/0 array for whether overlaps pass minimum linear extent
        """

        delta_mat = copy.deepcopy(delta_mat)
        npix_mat = copy.deepcopy(npix_mat)
        rms_mat = copy.deepcopy(rms_mat)
        lin_size_mat = copy.deepcopy(lin_size_mat)

        ns = delta_mat.shape[0]

        # Remove things with weights less than min_area_percent of the average weight
        avg_npix_val = np.nanmean(npix_mat[npix_mat != 0])
        small_area_idx = np.where(
            np.logical_and(
                npix_mat < self.min_area_percent * avg_npix_val, delta_mat != 0
            )
        )
        delta_mat[small_area_idx] = 0
        npix_mat[small_area_idx] = 0
        rms_mat[small_area_idx] = 0

        # Remove things that haven't passed the small area test
        delta_mat[lin_size_mat == 0] = 0
        npix_mat[lin_size_mat == 0] = 0
        rms_mat[lin_size_mat == 0] = 0

        # Remove fits with RMS values twice that of the mean
        avg_rms_val = np.nanmean(rms_mat[npix_mat != 0])
        sig_rms_val = np.nanstd(rms_mat[npix_mat != 0])
        rms_idx = np.where(rms_mat > avg_rms_val + self.rms_sig_limit * sig_rms_val)
        delta_mat[rms_idx] = 0
        npix_mat[rms_idx] = 0
        rms_mat[rms_idx] = 0

        # Make sure we've got rid of everything we need to
        delta_mat[npix_mat == 0] = 0
        delta_mat[rms_mat == 0] = 0

        # Create weight matrix
        if self.weight_method == "equal":
            # Weight evenly
            weight = np.ones_like(delta_mat)
            weight[delta_mat == 0] = 0
        elif self.weight_method == "npix":
            # Weight by straight number of pixels
            weight = 0.5 * (npix_mat + npix_mat.T)
        elif self.weight_method == "rms":
            # Weight by inverse variance of the fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weight = 0.5 * (rms_mat + rms_mat.T)
                weight = weight**-2
                weight[~np.isfinite(weight)] = 0
        else:
            raise ValueError(f"weight_method {self.weight_method} not known")

        neq = 0
        for i in range(ns):
            for j in range(i + 1, ns):
                if delta_mat[i, j] != 0 and npix_mat[i, j] > 0 and weight[i, j] > 0:
                    neq += 1

        # Create arrays for coefficients and free terms
        k = np.zeros((neq, ns), dtype=float)
        f = np.zeros(neq, dtype=float)
        invalid = ns * [True]

        # Process intersections between the rest of the images
        ieq = 0
        for i in range(0, ns):
            for j in range(i + 1, ns):
                # Only pull out valid intersections
                if delta_mat[i, j] != 0 and weight[i, j] > 0 and npix_mat[i, j] > 0:
                    k[ieq, i] = weight[i, j]
                    k[ieq, j] = -weight[i, j]

                    f[ieq] = weight[i, j] * delta_mat[i, j]
                    invalid[i] = False
                    invalid[j] = False
                    ieq += 1

        rank = np.linalg.matrix_rank(k, 1.0e-12)

        if rank < ns - 1:
            logging.warning(
                f"There are more unknown sky values ({ns}) to be solved for"
            )
            logging.warning(
                "than there are independent equations available "
                f"(matrix rank={rank})."
            )
            logging.warning("Sky matching (delta) values will be computed only for")
            logging.warning("a subset (or more independent subsets) of input images.")

        inv_k = np.linalg.pinv(k, rcond=1.0e-12)

        deltas = np.dot(inv_k, f)
        deltas[np.asarray(invalid, dtype=bool)] = 0

        return deltas

    def get_reproject(
        self,
        file,
        optimal_wcs,
        optimal_shape,
        stacked_image=False,
    ):
        """Reproject files, maintaining list structure

        Args:
            file: List or single file to reproject
            optimal_wcs: WCS to reproject to
            optimal_shape: output array shape for the WCS
            stacked_image (bool): Whether this is a stacked image or not. Defaults to False
        """

        if isinstance(file, list):
            file_reproj = [
                reproject_image(
                    i,
                    optimal_wcs=optimal_wcs,
                    optimal_shape=optimal_shape,
                    do_sigma_clip=self.do_sigma_clip,
                    stacked_image=stacked_image,
                )
                for i in file
            ]
        else:
            file_reproj = reproject_image(
                file,
                optimal_wcs=optimal_wcs,
                optimal_shape=optimal_shape,
                do_sigma_clip=self.do_sigma_clip,
                stacked_image=stacked_image,
            )

        return file_reproj

    def parallel_get_reproject(
        self,
        idx,
        files,
        optimal_wcs,
        optimal_shape,
        stacked_image=False,
    ):
        """Light function to parallelise get_dither_reproject

        Args:
            idx: File idx to reproject
            files: Full file list
            optimal_wcs: Optimal WCS for input stack of images
            optimal_shape: Optimal shape for input stack of images
            stacked_image: Stacked image or not? Defaults to False
        """

        dither_reproj = self.get_reproject(
            file=files[idx],
            optimal_wcs=optimal_wcs,
            optimal_shape=optimal_shape,
            stacked_image=stacked_image,
        )

        return idx, dither_reproj

    def get_plot_name(
        self,
        files1,
        files2,
    ):
        """Make a plot name from list of files for level matching

        Args:
            files1: First list of files
            files2: Second list of files
        """
        if isinstance(files1, list):
            files1_name_split = os.path.split(files1[0])[-1].split("_")

            # Since these should be dither groups, blank out the dither
            files1_name_split[2] = "XXXXX"
        else:
            files1_name_split = os.path.split(files1)[-1].split("_")
        plot_to_name = "_".join(files1_name_split[:-1])

        if isinstance(files2, list):
            files2_name_split = os.path.split(files2[0])[-1].split("_")

            # Since these should be dither groups, blank out the dither
            files2_name_split[2] = "XXXXX"
        else:
            files2_name_split = os.path.split(files2)[-1].split("_")
        plot_from_name = "_".join(files2_name_split[:-1])

        plot_name = os.path.join(
            self.plot_dir,
            f"{plot_from_name}_to_{plot_to_name}",
        )

        return plot_name
