import copy
import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
import warnings
from functools import partial

import astropy.units as u
import crds
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from jwst.flatfield.flat_field import do_correction
from mpl_toolkits.axes_grid1 import make_axes_locatable
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from reproject.mosaicking.background import determine_offset_matrix, solve_corrections_sgd
from scipy.ndimage import uniform_filter, median_filter
from stdatamodels.jwst import datamodels
from tqdm import tqdm

from ..utils import get_dq_bit_mask, make_source_mask, reproject_image, level_data

matplotlib.use("agg")
log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

ALLOWED_WEIGHT_METHODS = ["mean", "median", "sigma_clip"]
ALLOWED_WEIGHT_TYPES = ["exptime", "ivm", "equal"]


def get_rotation_angle(wcs):
    """Get rotation from a WCS instance

    Args:
        wcs: WCS instance
    """

    pc = np.dot(np.diag(wcs.wcs.get_cdelt()), wcs.wcs.get_pc())

    north = np.arctan2(-pc[0, 1],
                       pc[0, 0],
                       )
    angle = (north * u.rad).to(u.deg).value

    return angle


def make_diagnostic_plot(
        plot_name,
        data,
        stripes,
        figsize=(9, 4),
):
    """Make a diagnostic plot to show the destriping

    Args:
        plot_name: Output name for plot
        data: Original data
        stripes: Stripe model
        figsize: Size for the figure. Defaults to (9, 4)
    """

    plt.figure(figsize=figsize)

    n_rows = 1
    n_cols = 3

    vmin_data, vmax_data = np.nanpercentile(data, [10, 90])
    vmin_stripes, vmax_stripes = np.nanpercentile(stripes, [1, 99])

    # Plot the uncorrected data
    ax = plt.subplot(n_rows, n_cols, 1)

    im = plt.imshow(
        data,
        origin="lower",
        interpolation="nearest",
        vmin=vmin_data,
        vmax=vmax_data,
    )
    plt.axis("off")

    plt.text(
        0.05,
        0.95,
        "Uncorr. data",
        ha="left",
        va="top",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
        transform=ax.transAxes,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0)

    plt.colorbar(im, cax=cax, label="MJy/sr", orientation="horizontal")

    # Plot the stripes model
    ax = plt.subplot(n_rows, n_cols, 2)

    im = plt.imshow(
        stripes,
        origin="lower",
        interpolation="nearest",
        vmin=vmin_stripes,
        vmax=vmax_stripes,
    )
    plt.axis("off")

    plt.text(
        0.05,
        0.95,
        "Noise model",
        ha="left",
        va="top",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
        transform=ax.transAxes,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0)

    plt.colorbar(im, cax=cax, label="MJy/sr", orientation="horizontal")

    # And finally, the corrected data
    ax = plt.subplot(n_rows, n_cols, 3)

    im = plt.imshow(
        data - stripes,
        origin="lower",
        interpolation="nearest",
        vmin=vmin_data,
        vmax=vmax_data,
    )
    plt.axis("off")

    plt.text(
        0.05,
        0.95,
        "Corr. data",
        ha="left",
        va="top",
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
        transform=ax.transAxes,
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0)

    plt.colorbar(im, cax=cax, label="MJy/sr", orientation="horizontal")

    plt.subplots_adjust(wspace=0.01)

    plt.savefig(f"{plot_name}.png", bbox_inches="tight")
    plt.savefig(f"{plot_name}.pdf", bbox_inches="tight")

    plt.close()

    return True


def parallel_reproject_weight(
        idx,
        files,
        optimal_wcs,
        optimal_shape,
        weight_type="exptime",
        do_level_data=True,
):
    """Function to parallelise reprojecting with associated weights

    Args:
        idx: File idx to reproject
        files: Full stack of files
        optimal_wcs: Optimal WCS for image stack
        optimal_shape: Optimal shape for image shape
        weight_type: How to weight the average image. Defaults
            to exptime, the exposure time
        do_level_data: Whether to level data or not. Defaults to True
    """

    file = files[idx]

    data_array = reproject_image(
        file,
        optimal_wcs=optimal_wcs,
        optimal_shape=optimal_shape,
        do_level_data=do_level_data,
    )
    # Set any bad data to 0
    data_array.array[np.isnan(data_array.array)] = 0

    if weight_type == "exptime":
        with datamodels.open(file) as model:
            # Create array of exposure time
            exptime = model.meta.exposure.exposure_time
        del model
        weight_array = copy.deepcopy(data_array)
        weight_array.array[np.isfinite(weight_array.array)] = exptime
        weight_array.array[data_array.array == 0] = 0

    elif weight_type == "ivm":
        # Reproject the VAR_RNOISE array and take inverse
        weight_array = reproject_image(
            file,
            optimal_wcs=optimal_wcs,
            optimal_shape=optimal_shape,
            hdu_type="var_rnoise",
        )
        weight_array.array = weight_array.array ** -1
        weight_array.array[np.isnan(weight_array.array)] = 0

    elif weight_type == "equal":

        weight_array = copy.deepcopy(data_array)
        weight_array.array[np.isfinite(weight_array.array)] = 1
        weight_array.array[data_array.array == 0] = 0

    else:
        raise ValueError(f"weight_type should be one of {ALLOWED_WEIGHT_TYPES}")

    return idx, (file, data_array, weight_array)


class MultiTileDestripeStep:
    def __init__(
            self,
            in_dir,
            out_dir,
            step_ext,
            procs,
            apply_to_unflat=False,
            do_convergence=False,
            convergence_sigma=1,
            convergence_max_iterations=5,
            weight_method="mean",
            weight_type="ivm",
            do_level_match=False,
            quadrants=True,
            min_mask_frac=0.2,
            do_vertical_subtraction=False,
            do_large_scale=True,
            large_scale_filter_scale=None,
            large_scale_filter_extend_mode="reflect",
            sigma=3,
            dilate_size=7,
            maxiters=None,
            overwrite=False,
    ):
        """Subtracts large-scale stripes using dither information

        Create a weighted average image, then do a sigma-clipped median along (optionally)
        columns and rows (optionally by quadrants), after optionally smoothing the stacked
        image to attempt to remove persistent large-scale ripples.

        If you see clear oversubtraction in the data, you should set do_large_scale to False.
        In most cases, it appears to work well but there may be some edge cases where it doesn't
        work well.

        Args:
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel
            apply_to_unflat: If True, will undo the flat-fielding
                before applying the stripe model, and then
                reapply it. Defaults to False
            do_convergence: Whether to loop this iteratively
                until convergence, or just do a single run.
                Defaults to False
            convergence_sigma: Maximum sigma difference to decide
                if the iterative loop has converged. Defaults to 1
            convergence_max_iterations: Maximum number of iterations
                to run. Defaults to 5
            weight_type: Weighting method for stacking the image.
                Should be one of 'mean', 'median', 'sigma_clip'. Defaults
                to 'mean'
            weight_type: How to weight the stacked image.
                Defaults to 'ivm', inverse readnoise
            do_level_match: Whether to do a simple match between tiles. Should be set
                to False if this is run after level_match_step. Defaults to False
            quadrants: Whether to split up stripes per-amplifier. Defaults to True
            min_mask_frac: Minimum fraction of unmasked data in quadrants to calculate a median.
                Defaults to 0.2 (i.e. 20% unmasked)
            do_vertical_subtraction: Whether to also do a step of vertical stripe
                subtraction. Defaults to False
            do_large_scale: Whether to do filtering to try and remove large,
                consistent ripples between data. Defaults to True
            large_scale_filter_scale: Factor by which we smooth for large scale persistent
                ripple removal. Defaults to None, which will use a scale ~10% of the data shape
            large_scale_filter_extend_mode: How to extend values in the filter beyond
                array edge. Default is "reflect". See the specific docs for more info
            sigma: sigma value for sigma-clipped statistics. Defaults to 3
            dilate_size: Dilation size for mask creation. Defaults to 7
            maxiters: Maximum number of sigma-clipping iterations. Defaults to None
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        if weight_method not in ALLOWED_WEIGHT_METHODS:
            raise ValueError(
                f"weight_method should be one of {ALLOWED_WEIGHT_METHODS}, not {weight_method}"
            )
        if weight_type not in ALLOWED_WEIGHT_TYPES:
            raise ValueError(
                f"weight_type should be one of {ALLOWED_WEIGHT_TYPES}, not {weight_type}"
            )

        if weight_method in ["median", "sigma_clip"]:
            log.info(f"Using {weight_method} for creating average image, will not use weighting")

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.plot_dir = os.path.join(
            self.out_dir,
            "plots",
        )
        self.step_ext = step_ext
        self.procs = procs

        self.apply_to_unflat = apply_to_unflat
        self.do_convergence = do_convergence
        self.convergence_sigma = convergence_sigma
        self.convergence_max_iterations = convergence_max_iterations
        self.weight_method = weight_method
        self.weight_type = weight_type
        self.do_level_match = do_level_match
        self.quadrants = quadrants
        self.min_mask_frac = min_mask_frac
        self.do_large_scale = do_large_scale
        self.do_vertical_subtraction = do_vertical_subtraction
        self.large_scale_filter_scale = large_scale_filter_scale
        self.large_scale_filter_extend_mode = large_scale_filter_extend_mode
        self.sigma = sigma
        self.dilate_size = dilate_size
        self.maxiters = maxiters
        self.overwrite = overwrite

        self.files_reproj = None
        self.data_avg = None
        self.data_avg_smooth = None
        self.data_avg_mask = None
        self.optimal_wcs = None
        self.optimal_shape = None

    def do_step(self):
        """Run multi-tile destriping"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "multi_tile_destripe_step_complete.txt",
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

        # files = files[:16:4]

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(files)])

        # Get out the optimal WCS, since we only need to calculate this once
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimal_wcs, optimal_shape = find_optimal_celestial_wcs(files,
                                                                    hdu_in="SCI",
                                                                    auto_rotate=True,
                                                                    )

        self.optimal_wcs = optimal_wcs
        self.optimal_shape = optimal_shape

        converged = False
        iteration = 1

        while not converged:
            if self.do_convergence:
                log.info(f"Performing iteration {iteration}")

            # Create weighted images
            success = self.weighted_reproject_image(
                files,
                procs=procs,
                do_large_scale=False,
            )
            if not success:
                log.warning("Error in creating reproject stack")
                return False

            stripe_sigma = self.run_multi_tile_destripe(
                procs=procs,
                iteration=iteration,
                do_large_scale=False,
            )

            # Use the output files as potential further input
            files = glob.glob(
                os.path.join(
                    self.out_dir,
                    f"*_{self.step_ext}.fits",
                )
            )
            files.sort()

            # If doing large-scale we repeat, but using the output files
            if self.do_large_scale:

                log.info("Now doing large-scale smoothing for destriping")

                # Create weighted images
                success = self.weighted_reproject_image(
                    files,
                    procs=procs,
                    do_large_scale=True,
                )
                if not success:
                    log.warning("Error in creating reproject stack")
                    return False

                stripe_sigma = self.run_multi_tile_destripe(
                    procs=procs,
                    iteration=iteration,
                    do_large_scale=True,
                )

            # If we're not iterating, then say we've converged
            if not self.do_convergence:
                converged = True
            else:
                if not np.all(stripe_sigma < self.convergence_sigma):
                    if iteration < self.convergence_max_iterations:
                        log.info("Destriping not converged. Continuing")
                    else:
                        log.info(
                            "Destriping not converged but max iterations reached. Final stripe sigma values:"
                        )
                        for i, file in enumerate(files):
                            log.info(f"{os.path.split(file)[-1]}, {stripe_sigma[i]}")
                        converged = True
                else:
                    log.info("Convergence reached! Final stripe sigma values:")
                    for i, file in enumerate(files):
                        log.info(f"{os.path.split(file)[-1]}: {stripe_sigma[i]}")
                    converged = True

            iteration += 1

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def weighted_reproject_image(
            self,
            files,
            procs=1,
            do_large_scale=False,
    ):
        """Get reprojected images (and weights)

        Args:
            files (list): Files to reproject
            procs (int): Number of processes to use. Defaults to 1.
            do_large_scale: Is this a large-scale smoothed subtraction? Defaults to False
        """

        data_reproj = [None] * len(files)
        weight_reproj = [None] * len(files)
        files_reproj = [None] * len(files)

        log.info(f"Reprojecting images (and weights)")

        with mp.get_context("fork").Pool(procs) as pool:
            for i, result in tqdm(
                    pool.imap_unordered(
                        partial(
                            parallel_reproject_weight,
                            files=files,
                            optimal_wcs=self.optimal_wcs,
                            optimal_shape=self.optimal_shape,
                            weight_type=self.weight_type,
                            do_level_data=True,
                        ),
                        range(len(files)),
                    ),
                    total=len(files),
                    desc="Weighted reprojects",
                    ascii=True,
            ):
                files_reproj[i] = result[0]
                data_reproj[i] = result[1]
                weight_reproj[i] = result[2]

            pool.close()
            pool.join()
            gc.collect()

        self.files_reproj = files_reproj

        # Create the weighted average image here

        log.info("Creating average image")

        self.data_avg = self.create_weighted_avg_image(data_reproj,
                                                       weight_reproj,
                                                       )

        # Smooth out the data for large-scale correction
        if do_large_scale:

            log.info("Figuring out rotation between input images and stack")

            ref_rot = get_rotation_angle(self.optimal_wcs)
            indiv_rots = []

            for file in self.files_reproj:
                with datamodels.open(file) as im:
                    wcs = im.meta.wcs.to_fits_sip()
                    w = WCS(wcs)

                    indiv_rots.append(get_rotation_angle(w))

                    # Also get the filter scale, if necessary
                    if self.large_scale_filter_scale is None:
                        large_scale_filter_scale = im.data.shape[0] // 10
                        if large_scale_filter_scale % 2 == 0:
                            large_scale_filter_scale -= 1
                        self.large_scale_filter_scale = large_scale_filter_scale

            indiv_rots = np.array(indiv_rots)

            # Look for big differences between tiles
            internal_diff = np.abs(indiv_rots - indiv_rots[0])
            # Account for the quadrants of this space
            internal_diff[internal_diff >= 180] -= 180
            internal_diff = np.abs(internal_diff)
            internal_diff[internal_diff >= 90] -= 180
            internal_diff = np.abs(internal_diff)

            # And differences wrt the reference WCS
            ref_diff = np.abs(indiv_rots - ref_rot)
            # Account for the quadrants of this space
            ref_diff[ref_diff >= 180] -= 180
            ref_diff = np.abs(ref_diff)
            ref_diff[ref_diff >= 90] -= 180
            ref_diff = np.abs(ref_diff)

            # First case, we have a weird mix of rotations. In which case direction should be None
            if not np.all(internal_diff < 10):
                raise ValueError("Input images have a variety of rotations. Have not encountered this before")
                direction = None
                log.info("Input images have a variety of rotations. Defaulting to smoothing over both axes")

            # Second case, they're all similar to the reference rotation, so stripes are horizontal
            elif np.all(ref_diff < 10):
                direction = "horizontal"
                log.info("Stacked image is at same rotation as input images")

            # Third case, they're perpendicular to the reference rotation, so stripes are vertical
            elif np.all(np.logical_and(80 < ref_diff, ref_diff < 100)):
                direction = "vertical"
                log.info("Stacked image is perpendicular to input images")

            # Final case, they're aligned but not along any particular axis in the image. In this case,
            # direction should be None
            else:
                direction = None
                log.info("Stacked image does not align over a particular axis. Smoothing over both axes")

            log.info("Creating smoothed image")

            self.data_avg_smooth, self.data_avg_mask = self.get_data_avg_smooth(direction=direction)

        return True

    def run_multi_tile_destripe(
            self,
            procs=1,
            iteration=1,
            do_large_scale=False,
    ):
        """Wrap parallelism around the multi-tile destriping

        Args:
            procs: Number of parallel processes. Defaults to 1
            iteration: What iteration are we on? Defaults to 1
            do_large_scale: Is this a large-scale smoothed subtraction? Defaults to False
        """

        log.info("Running multi-tile destripe")

        with mp.get_context("fork").Pool(procs) as pool:
            results = [None] * len(self.files_reproj)

            for i, result in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_multi_tile_destripe,
                            iteration=iteration,
                            do_large_scale=do_large_scale,
                        ),
                        range(len(self.files_reproj)),
                    ),
                    total=len(self.files_reproj),
                    ascii=True,
                    desc="Multi-tile destriping",
            ):
                results[i] = result

            pool.close()
            pool.join()
            gc.collect()

        stripe_sigma = np.zeros(len(self.files_reproj))

        for result in results:
            if result is not None:
                in_file = result[0]
                stripes = result[1]

                short_file = os.path.split(in_file)[-1]
                out_file = os.path.join(
                    self.out_dir,
                    short_file,
                )

                with datamodels.open(in_file) as im:

                    zero_idx = np.where(im.data == 0)
                    nan_idx = np.where(np.isnan(im.data))

                    if self.apply_to_unflat:

                        # Get CRDS context
                        try:
                            crds_context = os.environ["CRDS_CONTEXT"]
                        except KeyError:
                            crds_context = crds.get_default_context()

                        crds_dict = {
                            "INSTRUME": "NIRCAM",
                            "DETECTOR": im.meta.instrument.detector,
                            "FILTER": im.meta.instrument.filter,
                            "PUPIL": im.meta.instrument.pupil,
                            "DATE-OBS": im.meta.observation.date,
                            "TIME-OBS": im.meta.observation.time,
                        }
                        flats = crds.getreferences(crds_dict, reftypes=["flat"], context=crds_context)
                        flatfile = flats["flat"]

                        with datamodels.FlatModel(flatfile) as flat:

                            flat_inverse = copy.deepcopy(flat)
                            flat_inverse.data = flat_inverse.data ** -1

                            # First, unapply the flat fielding to the image and subtract stripes
                            unflattened_im, _ = do_correction(im, flat_inverse)
                            unflattened_im.data -= stripes

                            # Reflatten and save into the original data array
                            reflattened_im, _ = do_correction(unflattened_im, flat)
                            im.data = copy.deepcopy(reflattened_im.data)

                    else:
                        im.data -= stripes

                    # If we're not in subarray mode, here we want to level out between amplifiers
                    # for safety
                    if "sub" not in im.meta.subarray.name.lower():
                        im.data = level_data(im)

                    im.data[zero_idx] = 0
                    im.data[nan_idx] = np.nan

                    im.save(out_file)

                    # Find the right index, since multiprocessing doesn't
                    # preserve the order necessarily
                    idx = self.files_reproj.index(in_file)

                    # Calculate the maximum sigma-values for the stripes wrt error
                    err = copy.deepcopy(im.err)
                    err[err == 0] = np.nan
                    stripe_sigma[idx] = np.abs(np.nanmax(stripes / im.err))

                del im

        gc.collect()

        return stripe_sigma

    def parallel_multi_tile_destripe(
            self,
            idx,
            iteration=1,
            do_large_scale=False,
    ):
        """Function to parallelise up multi-tile destriping

        Args:
            idx: Index of file to be destriped
            iteration: What iteration are we on? Defaults to 1
            do_large_scale: Is this a large-scale smoothed subtraction? Defaults to False
        """

        file = self.files_reproj[idx]

        result = self.multi_tile_destripe(
            file,
            iteration=iteration,
            do_large_scale=do_large_scale,
        )
        return idx, result

    def create_weighted_avg_image(
            self,
            data,
            weights,

    ):
        """Create an average image from a bunch of reprojected ones

        Args:
            data: List of data arrays
            weights: List of weights. Should be same length as data
        """

        # Start by calculating corrections to match between tiles, if not
        # already done
        if self.do_level_match:
            offset_matrix = determine_offset_matrix(data)
            corrections = solve_corrections_sgd(offset_matrix)
            for array, correction in zip(data, corrections):
                zero_idx = np.where(array.array == 0)
                array.array -= correction
                array.array[zero_idx] = 0

        # First, put the original image in
        if self.weight_method == "mean":
            output_array = np.zeros(self.optimal_shape)
        elif self.weight_method in ["median", "sigma_clip"]:
            output_array = np.zeros([self.optimal_shape[0], self.optimal_shape[1], len(data)])
        else:
            raise ValueError(f"weight_method should be one of {ALLOWED_WEIGHT_METHODS}")

        output_weights = np.zeros_like(output_array)

        for i, (array, weight) in enumerate(zip(data, weights)):

            # Put the reprojected data into the array. This will be different depending on
            # the weight method
            if self.weight_method == "mean":
                output_array[array.view_in_original_array] += array.array * weight.array
                output_weights[weight.view_in_original_array] += weight.array
            elif self.weight_method in ["median", "sigma_clip"]:
                output_array[array.view_in_original_array[0], array.view_in_original_array[1], i] = array.array
                output_weights[weight.view_in_original_array[0], weight.view_in_original_array[1], i] = weight.array
            else:
                raise ValueError(
                    f"weight_method should be one of {ALLOWED_WEIGHT_METHODS}"
                )

        output_array[output_weights == 0] = np.nan

        # Now we can calculate the average. For the mean, this is weighted
        if self.weight_method == "mean":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_avg = output_array / output_weights

        # For the median, ignore weights (apart from 0s)
        elif self.weight_method == "median":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_avg = np.nanmedian(data, axis=-1)

        # Sigma-clipped median (this will ignore weights)
        elif self.weight_method == "sigma_clip":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data_avg = sigma_clipped_stats(output_array,
                                               sigma=self.sigma,
                                               maxiters=self.maxiters,
                                               axis=-1,
                                               )[1]

        else:
            raise ValueError(f"weight_method should be one of {ALLOWED_WEIGHT_METHODS}")

        data_avg[data_avg == 0] = np.nan
        data_avg[~np.isfinite(data_avg)] = np.nan

        return data_avg

    def get_data_avg_smooth(self,
                            direction=None,
                            ):
        """Filter data with a large scale filter

        Will either perform a large-scale median filter over a specific axis,
        or a mean filter over all axes. Also creates a mask

        Args:
            direction: Direction to smooth over, either "horizontal", "vertical", or None.
                Defaults to None
        """

        data_avg = copy.deepcopy(self.data_avg)
        nan_idx = np.where(np.isnan(data_avg))

        if direction in ["horizontal", None]:
            interp_order = [1, 0]
        elif direction == "vertical":
            interp_order = [0, 1]
        else:
            raise ValueError("direction should be horizontal, vertical, or None")

        # Extrapolate over axes in order based on where the stripes are
        for order in interp_order:
            for ax in range(data_avg.shape[order]):

                if order == 0:
                    data_ax = copy.deepcopy(data_avg[ax, :])
                else:
                    data_ax = copy.deepcopy(data_avg[:, ax])

                mask = np.isnan(data_ax)

                # Only interp if we have a) some NaNs but not b) all NaNs
                if 0 < np.sum(mask) < len(data_ax):
                    data_ax[mask] = np.interp(np.flatnonzero(mask),
                                              np.flatnonzero(~mask),
                                              data_ax[~mask],
                                              )

                if order == 0:
                    data_avg[ax, :] = copy.deepcopy(data_ax)
                else:
                    data_avg[:, ax] = copy.deepcopy(data_ax)

        log.info(f"Smoothing with a filter scale of {self.large_scale_filter_scale}")

        if direction is None:
            data_smooth = uniform_filter(data_avg,
                                         size=self.large_scale_filter_scale,
                                         mode=self.large_scale_filter_extend_mode,
                                         )
        else:

            data_smooth = np.zeros_like(data_avg)

            if direction == "horizontal":

                for row in range(data_avg.shape[1]):
                    col = data_avg[:, row]
                    col_smooth = median_filter(col,
                                               size=self.large_scale_filter_scale,
                                               mode=self.large_scale_filter_extend_mode,
                                               )
                    data_smooth[:, row] = copy.deepcopy(col_smooth)

            elif direction == "vertical":

                for col in range(data_avg.shape[0]):
                    row = data_avg[col, :]
                    row_smooth = median_filter(row,
                                               size=self.large_scale_filter_scale,
                                               mode=self.large_scale_filter_extend_mode,
                                               )
                    data_smooth[col, :] = copy.deepcopy(row_smooth)

            else:
                raise ValueError("direction should be one of horizontal, vertical")

        data_smooth[nan_idx] = np.nan

        mask = self.get_mask(self.data_avg - data_smooth)

        return data_smooth, mask

    def get_mask(self,
                 data,
                 ):
        """Create positive/negative mask"""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask_pos = make_source_mask(
                data,
                nsigma=self.sigma,
                dilate_size=self.dilate_size,
                sigclip_iters=self.maxiters,
            )
            mask_neg = make_source_mask(
                -data,
                mask=mask_pos,
                nsigma=self.sigma,
                dilate_size=self.dilate_size,
                sigclip_iters=self.maxiters,
            )
        mask = mask_pos | mask_neg

        return mask

    def multi_tile_destripe(
            self,
            file,
            iteration=1,
            do_large_scale=False,
    ):
        """Do a row-by-row, column-by-column data subtraction using other dither information

        Reproject average image, optionally remove persistent large-scale stripes, then do a sigma-clipped
        median along columns and rows (optionally by quadrants), and finally a smoothed clip along
        rows after boxcar filtering to remove persistent large-scale ripples in the data

        Args:
            file (str): File to correct
            iteration: What iteration are we on? Defaults to 1
            do_large_scale: Is this a large-scale smoothed subtraction? Defaults to False
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with datamodels.open(file) as model:
                file_name = model.meta.filename

                quadrants = copy.deepcopy(self.quadrants)

                # If we're in subarray mode or doing large-scale, turn off quadrants
                if "sub" in model.meta.subarray.name.lower():
                    quadrants = False
                if do_large_scale:
                    quadrants = False

                # If we're not in subarray mode, level everything out
                else:
                    model.data = level_data(model)

                dq_bit_mask = get_dq_bit_mask(model.dq)

                # Pull out data and DQ mask
                data = copy.deepcopy(model.data)
                data[dq_bit_mask != 0] = np.nan

                wcs = model.meta.wcs.to_fits_sip()
            del model

            # Reproject the average image
            data_avg = reproject_interp(
                (self.data_avg, self.optimal_wcs),
                wcs,
                return_footprint=False,
            )

            # If we're also attempting to remove large-scale ripples, we filter the average
            # data here and correct the average image
            if do_large_scale:

                # Reproject the smoothed data
                data_avg_smooth = reproject_interp(
                    (self.data_avg_smooth, self.optimal_wcs),
                    wcs,
                    return_footprint=False,
                )

                diff_smooth = data_avg - data_avg_smooth

                # Also reproject the mask, casting to bool
                mask_smooth = reproject_interp(
                    (self.data_avg_mask, self.optimal_wcs),
                    wcs,
                    order='nearest-neighbor',
                    return_footprint=False,
                )
                mask_smooth = np.array(mask_smooth, dtype=bool)

                # Get the low-level stripes left in the data
                stripes_smooth = sigma_clipped_stats(
                    diff_smooth,
                    mask=mask_smooth,
                    sigma=self.sigma,
                    maxiters=self.maxiters,
                    axis=1,
                )[1]

                mask = np.isnan(stripes_smooth)
                # Only interp if we have a) some NaNs but not b) all NaNs
                if 0 < np.sum(mask) < len(stripes_smooth):
                    stripes_smooth[mask] = np.interp(np.flatnonzero(mask),
                                                     np.flatnonzero(~mask),
                                                     stripes_smooth[~mask],
                                                     )

                data_avg = data_avg - stripes_smooth[:, np.newaxis]

            diff = data - data_avg
            diff -= np.nanmedian(diff)

            stripes_arr = np.zeros_like(diff)

            mask_diff = self.get_mask(diff)

            if self.do_vertical_subtraction:
                # First, subtract the y
                stripes_y = sigma_clipped_stats(
                    diff - stripes_arr,
                    mask=mask_diff,
                    sigma=self.sigma,
                    maxiters=self.maxiters,
                    axis=0,
                )[1]

                # Centre around 0, replace NaNs with nearest value
                stripes_y -= np.nanmedian(stripes_y)

                mask = np.isnan(stripes_y)
                # Only interp if we have a) some NaNs but not b) all NaNs
                if 0 < np.sum(mask) < len(stripes_y):
                    stripes_y[mask] = np.interp(np.flatnonzero(mask),
                                                np.flatnonzero(~mask),
                                                stripes_y[~mask],
                                                )

                stripes_arr += stripes_y[np.newaxis, :]

            stripes_x_2d = np.zeros_like(stripes_arr)

            # Sigma-clip the diff across the whole image
            stripes_x_full = sigma_clipped_stats(
                diff - stripes_arr,
                mask=mask_diff,
                sigma=self.sigma,
                maxiters=self.maxiters,
                axis=1,
            )[1]
            stripes_x_full[stripes_x_full == 0] = np.nan

            if quadrants:
                quadrant_size = stripes_arr.shape[1] // 4

                for quadrant in range(4):
                    idx_slice = slice(
                        quadrant * quadrant_size, (quadrant + 1) * quadrant_size
                    )

                    # Sigma-clip the diff
                    diff_quadrants = (
                            diff[:, idx_slice] - stripes_arr[:, idx_slice]
                    )
                    mask_quadrants = mask_diff[:, idx_slice]
                    stripes_x = sigma_clipped_stats(
                        diff_quadrants,
                        mask=mask_quadrants,
                        sigma=self.sigma,
                        maxiters=self.maxiters,
                        axis=1,
                    )[1]
                    stripes_x[stripes_x == 0] = np.nan

                    mask_sum = np.nansum(~np.asarray(mask_quadrants, dtype=bool), axis=1)
                    too_masked_idx = np.where(mask_sum < quadrant_size * self.min_mask_frac)

                    # For anything with less than the requisite amount of unmasked pixels, fall
                    # back to the full row median
                    stripes_x[too_masked_idx] = stripes_x_full[too_masked_idx]

                    # Replace NaNs with nearest values
                    mask = np.isnan(stripes_x)
                    # Only interp if we have a) some NaNs but not b) all NaNs
                    if 0 < np.sum(mask) < len(stripes_x):
                        stripes_x[mask] = np.interp(np.flatnonzero(mask),
                                                    np.flatnonzero(~mask),
                                                    stripes_x[~mask],
                                                    )

                    # Centre around 0, since we've corrected for steps between amplifiers
                    stripes_x -= np.nanmedian(stripes_x)

                    stripes_x_2d[:, idx_slice] += stripes_x[:, np.newaxis]

            else:

                # Centre around 0, replace NaNs with nearest values
                stripes_x_full -= np.nanmedian(stripes_x_full)

                mask = np.isnan(stripes_x_full)

                # Only interp if we have a) some NaNs but not b) all NaNs
                if 0 < np.sum(mask) < len(stripes_x_full):
                    stripes_x_full[mask] = np.interp(np.flatnonzero(mask),
                                                     np.flatnonzero(~mask),
                                                     stripes_x_full[~mask],
                                                     )

                stripes_x_2d += stripes_x_full[:, np.newaxis]

            # Centre around 0 one last time
            stripes_x_2d -= np.nanmedian(stripes_x_2d)

            stripes_arr += stripes_x_2d
            stripes_arr -= np.nanmedian(stripes_arr)

            # Make diagnostic plot. Use different names if
            # we're iterating
            suffix = "_dither_stripe_sub"
            if do_large_scale:
                suffix += "_large_scale"
            if self.do_convergence:
                suffix += f"_iter_{iteration}"

            plot_name = os.path.join(
                self.plot_dir,
                file_name.replace(".fits", suffix),
            )

            make_diagnostic_plot(
                plot_name=plot_name,
                data=data,
                stripes=stripes_arr,
            )

        return file, stripes_arr
