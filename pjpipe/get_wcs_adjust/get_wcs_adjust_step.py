import copy
import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
import warnings
from functools import partial

import numpy as np
import spacepylot.alignment as align
from astropy.io import fits
from astropy.table import Table, QTable
from image_registration.fft_tools import shift
from jwst.datamodels import ModelContainer
from jwst.tweakreg import TweakRegStep
from reproject.mosaicking import find_optimal_celestial_wcs
from spacepylot.alignment_utilities import TranslationTransform
from stdatamodels.jwst import datamodels
from tqdm import tqdm

from ..utils import get_band_type, fwhms_pix, parse_parameter_dict, recursive_setattr, make_stacked_image, \
    reproject_image, get_pixscale

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

RAD_TO_ARCSEC = 3600 * np.rad2deg(1)

ALLOWED_METHODS = [
    "tweakreg",
    "cross_corr",
]

ALLOWED_REPROJECT_FUNCS = [
    "interp",
    "adaptive",
    "exact",
]


def write_visit_transforms(
        visit_transforms,
        out_file,
):
    """Write out table of WCS transforms

    Args:
        visit_transforms: Dictionary of transforms
            per visit
        out_file: Name for the output .toml file
    """
    log.info(f"Writing transforms")

    with open(out_file, "w+") as f:
        f.write("[wcs_adjust]\n\n")

        # Skip where we don't have anything
        if len(visit_transforms) == 0:
            log.info("No WCS adjusts found. Skipping")
            f.close()
            return True

        for visit in visit_transforms:
            # If we only have one shift value, take that, otherwise take the mean
            if len(visit_transforms[visit]["shift"].shape) == 1:
                shift = visit_transforms[visit]["shift"]
            else:
                shift = np.nanmean(visit_transforms[visit]["shift"], axis=0)

            # If we only have one matrix value, take that, otherwise take the mean
            if len(visit_transforms[visit]["matrix"].shape) == 2:
                matrix = visit_transforms[visit]["matrix"]
            else:
                matrix = np.nanmean(visit_transforms[visit]["matrix"], axis=-1)

            # Format these as nice strings and write out
            shift_str = [float(f"{s:.3f}") for s in shift]
            matrix_l1 = [float(f"{s:.3f}") for s in matrix[0]]
            matrix_l2 = [float(f"{s:.3f}") for s in matrix[1]]

            f.write(f"{visit}.shift = {shift_str}\n")
            f.write(f"{visit}.matrix = [\n\t{matrix_l1},\n\t{matrix_l2}\n]\n")

        f.write("\n")
        f.close()

    return True


class GetWCSAdjustStep:
    def __init__(
            self,
            directory,
            progress_dict,
            target,
            alignment_dir,
            procs=1,
            bands=None,
            method="tweakreg",
            alignment_catalogs=None,
            group_dithers=None,
            tweakreg_parameters=None,
            reproject_func="interp",
            overwrite=False,
    ):
        """Gets a table of WCS corrections to apply to visit groups

        Experience has shown that the relative JWST guide star uncertainty is very
        small, but there are significant absolute corrections between guide stars.
        Thus, we can use the same visit as a correction for all visits, for example
        using F1000W at F2100W where tweakreg doesn't work so well.

        Here, we take some template bands and loop over with tweakreg, writing out a table
        of shifts/matrices to apply to other bands. For multiple dithers etc., will take
        an average correction.

        Alternatively, we can take a cross-correlation approach. Here, we instead make a stacked
        image of the dithers for each mosaic tile, then loop over and cross-correlate to get a shift.
        This should work better than tweakreg for bands where there aren't many stars present, e.g.
        F770W.

        Args:
            directory: Directory of target
            progress_dict: The progress dictionary the pipeline builds up.
                This is used to figure out what subdirectories we should
                be looking in
            target: Target to consider
            alignment_dir: Directory for alignment catalogs
            bands: List of target bands to pull corrections out for
            method: Method to align images together. Can be "tweakreg" (default)
                or cross_corr. N.B. for "cross_corr", all dithers will be stacked by default
                to calculate the cross-correlation. This can also be a dictionary to distinguish
                between e.g. NIRCam and MIRI, like {'nircam': 'tweakreg', 'miri': 'cross_corr'}
            alignment_catalogs: Dictionary mapping targets to alignment catalogs
            procs: Number of processes to run in parallel. Defaults to 1
            group_dithers: Which band type (e.g. nircam) to group
                up dithers for and find a single correction. Defaults
                to None, which won't group up anything
            tweakreg_parameters: Dictionary of parameters to pass to tweakreg.
                Defaults to None, which will use observatory defaults
            reproject_func: Which reproject function to use. Defaults to 'interp',
                but can also be 'exact' or 'adaptive'
            overwrite: Whether to overwrite or not. Defaults to False
        """

        if bands is None:
            raise ValueError("Need some bands to get WCS adjustments")

        if isinstance(method, dict):
            for key in method:
                if method[key] not in ALLOWED_METHODS:
                    raise ValueError(f"method should be one of {ALLOWED_METHODS}")
        else:
            if method not in ALLOWED_METHODS:
                raise ValueError(f"method should be one of {ALLOWED_METHODS}")

        if reproject_func not in ALLOWED_REPROJECT_FUNCS:
            raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")

        if group_dithers is None:
            group_dithers = []
        if tweakreg_parameters is None:
            tweakreg_parameters = {}
        if alignment_catalogs is None:
            alignment_catalogs = {}

        # We should be using cal files, and saving them as wcs_adjust
        self.in_ext = "cal"
        self.out_ext = "wcs_adjust"

        self.directory = directory
        self.progress_dict = progress_dict
        self.target = target
        self.alignment_dir = alignment_dir
        self.procs = procs
        self.bands = bands
        self.method = method
        self.alignment_catalogs = alignment_catalogs
        self.group_dithers = group_dithers
        self.tweakreg_parameters = tweakreg_parameters
        self.reproject_func = reproject_func
        self.overwrite = overwrite

    def do_step(self):
        """Run the WCS adjust step"""

        step_complete_file = os.path.join(
            self.directory,
            "get_wcs_adjust_step_complete.txt",
        )
        out_file = os.path.join(self.directory, f"{self.target}_wcs_adjust.toml")

        if self.overwrite:
            if os.path.exists(out_file):
                os.remove(out_file)
            if os.path.exists(step_complete_file):
                os.remove(step_complete_file)

        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        out_dir = os.path.join(self.directory,
                               "get_wcs_adjust",
                               )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Perform the shifts
        for band_full in self.bands:

            if isinstance(self.method, str):
                method = copy.deepcopy(self.method)
            elif isinstance(self.method, dict):

                if "bgr" in band_full:
                    band = band_full.replace("_bgr", "")
                else:
                    band = copy.deepcopy(band_full)

                band_type = get_band_type(band)

                # Pull out the method, or fall back to tweakreg
                try:
                    method = self.method[band_type]
                except KeyError:
                    log.warning(f"Method not found for instrument {band_type}. Will default to tweakreg")
                    method = "tweakreg"

            else:
                raise ValueError("method should either be a string or dictionary")

            if method == "tweakreg":
                success = self.run_tweakreg(band_full=band_full, out_dir=out_dir)
            elif method == "cross_corr":
                success = self.run_cross_corr(band_full=band_full, out_dir=out_dir)
            else:
                raise ValueError(f"method should be one of {ALLOWED_METHODS}")

            if not success:
                log.warning("Failures detected in getting WCS adjustments")
                return False

        # Get the visit transforms
        visit_transforms = self.get_visit_transforms(
            in_dir=out_dir,
        )

        # Write out the visit transforms
        success = write_visit_transforms(
            visit_transforms,
            out_file,
        )

        if not success:
            log.warning("Failures detected in getting WCS adjustments")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_tweakreg(self,
                     band_full,
                     out_dir
                     ):
        """Run tweakreg to get shifts

        Args:
            band_full: Band to consider
            out_dir: Output directory to save files with shifts to
        """

        log.info(f"Running tweakreg for {band_full}")

        if "bgr" in band_full:
            band = band_full.replace("_bgr", "")
        else:
            band = copy.deepcopy(band_full)

        band_type = get_band_type(band)

        # Some various failure states
        if band_full not in self.progress_dict:
            log.warning(f"No data found for {band_full}. Skipping")
            return True
        if "dir" not in self.progress_dict[band_full]:
            log.warning(f"No files found for {band_full}. Skipping")
            return True
        if not self.progress_dict[band_full]["success"]:
            log.warning(f"Previous failures found for {band_full}. Skipping")
            return True

        band_dir = copy.deepcopy(self.progress_dict[band_full]["dir"])
        if not os.path.exists(band_dir):
            log.warning(f"Directory {band_dir} does not exist")
            return True

        fwhm_pix = fwhms_pix[band]

        in_files = glob.glob(
            os.path.join(
                band_dir,
                f"*_{self.in_ext}.fits",
            )
        )
        in_files.sort()
        input_models = [datamodels.open(in_file) for in_file in in_files]
        asn_file = ModelContainer(input_models)

        # Group up the dithers
        if band_type in self.group_dithers:
            for model in asn_file._models:
                model.meta.observation.exposure_number = "1"
                model.meta.group_id = ""

        # If we only have one group, this won't do anything so just skip
        if len(asn_file.models_grouped) == 1 and self.target not in self.alignment_catalogs:
            log.info(f"Only one group and no absolute alignment happening. Skipping")
            del input_models, asn_file
            return True

        tweakreg_config = TweakRegStep.get_config_from_reference(asn_file)
        tweakreg = TweakRegStep.from_config_section(tweakreg_config)
        tweakreg.output_dir = out_dir
        tweakreg.save_results = True
        tweakreg.suffix = self.out_ext
        tweakreg.kernel_fwhm = fwhm_pix * 2

        # Sort this into a format that tweakreg is happy with
        if self.target in self.alignment_catalogs:

            abs_ref_catalog = os.path.join(self.directory,
                                           f"{self.target}_ref_catalog.fits",
                                           )
            if not os.path.exists(abs_ref_catalog):
                in_catalog = os.path.join(self.alignment_dir,
                                          self.alignment_catalogs[self.target],
                                          )
                align_table = QTable.read(in_catalog, format="fits")
                abs_tab = Table()

                abs_tab["RA"] = align_table["ra"]
                abs_tab["DEC"] = align_table["dec"]
                abs_tab.write(abs_ref_catalog, overwrite=True)

            tweakreg.abs_refcat = abs_ref_catalog

        for tweakreg_key in self.tweakreg_parameters:
            value = parse_parameter_dict(
                self.tweakreg_parameters,
                tweakreg_key,
                band,
                self.target,
            )

            if value == "VAL_NOT_FOUND":
                continue

            recursive_setattr(tweakreg, tweakreg_key, value)

        tweakreg.run(asn_file)

        del input_models, asn_file

        return True

    def run_cross_corr(self,
                       band_full,
                       out_dir,
                       ):
        """Get transforms per-visit, using a cross-correlation between stacked dithers

        Args:
            band_full: Band to consider
            out_dir: Output directory to save files with shifts to
        """

        log.info(f"Running cross-correlation for {band_full}")

        # Some various failure states
        if band_full not in self.progress_dict:
            log.warning(f"No data found for {band_full}. Skipping")
            return True
        if "dir" not in self.progress_dict[band_full]:
            log.warning(f"No files found for {band_full}. Skipping")
            return True
        if not self.progress_dict[band_full]["success"]:
            log.warning(f"Previous failures found for {band_full}. Skipping")
            return True

        band_dir = copy.deepcopy(self.progress_dict[band_full]["dir"])
        if not os.path.exists(band_dir):
            log.warning(f"Directory {band_dir} does not exist")
            return True

        in_files = glob.glob(
            os.path.join(
                band_dir,
                f"*_{self.in_ext}.fits",
            )
        )
        in_files.sort()

        # Split these into dithers per-chip
        dithers = []
        for file in in_files:
            file_split = os.path.split(file)[-1].split("_")
            dithers.append("_".join(file_split[:2]) + "_*_" + file_split[-2])
        dithers = np.unique(dithers)
        dithers.sort()

        # If we only have one set of dithers, this won't do anything so just skip
        if len(dithers) == 1:
            log.info(f"Only one group. Skipping")
            return True

        # Create stacked images, so we can figure out the overlap between the dithers
        procs = np.nanmin([self.procs, len(dithers)])

        successes = self.make_stacked_images(dithers,
                                             in_dir=band_dir,
                                             out_dir=out_dir,
                                             procs=procs,
                                             )

        if not np.all(successes):
            log.warning("Failures detected making stacked images")
            return False

        # Find the stacked images we've just created. There might be other stuff in that directory so be careful!
        stacked_images = []
        for dither in dithers:
            stacked_images.extend(glob.glob(os.path.join(out_dir, f"{dither}*{self.in_ext}.fits")))

        # Reproject these to a common best WCS, which should be north-aligned
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimal_wcs, optimal_shape = find_optimal_celestial_wcs(
                stacked_images,
                hdu_in="SCI",
            )

        # Save the reprojected arrays to a dictionary, so it's easy to pull them out
        reproj_array_dict = {}
        for i, dither in enumerate(dithers):
            reproj_array = reproject_image(
                stacked_images[i],
                optimal_wcs=optimal_wcs,
                optimal_shape=optimal_shape,
                stacked_image=True,
                reproject_func=self.reproject_func,
            )

            reproj_array_dict[dither] = copy.deepcopy(reproj_array)

        # Now we'll start looping. Keep track of what we've matched or not
        shift_dict = {}

        unmatched_dict = {}
        for dither in dithers:
            unmatched_dict[dither] = {}
            unmatched_dict[dither]["max_overlap_pixels"] = 0
            unmatched_dict[dither]["max_overlap_dither"] = None

        log.info("Finding maximum overlaps and tiles with no overlaps")
        for i in range(len(dithers)):
            for j in range(i + 1, len(dithers)):
                diff = reproj_array_dict[dithers[j]] - reproj_array_dict[dithers[i]]
                overlap_pix = diff.footprint[diff.footprint != 0]

                n_overlap_pix = np.sum(overlap_pix)

                # These things are symmetric, so we only need to check one way
                if n_overlap_pix > unmatched_dict[dithers[i]]["max_overlap_pixels"]:
                    unmatched_dict[dithers[i]]["max_overlap_pixels"] = n_overlap_pix
                    unmatched_dict[dithers[j]]["max_overlap_pixels"] = n_overlap_pix

                    unmatched_dict[dithers[i]]["max_overlap_dither"] = dithers[j]
                    unmatched_dict[dithers[j]]["max_overlap_dither"] = dithers[i]

        all_overlap_pixels = np.array([unmatched_dict[dither]["max_overlap_pixels"] for dither in dithers])

        # Step 0: Find any tiles that don't overlap, give these a shift of [0,0]
        no_overlap_idx = np.where(all_overlap_pixels == 0)[0]

        for idx in no_overlap_idx:
            no_overlap_dither = dithers[idx]
            log.info(f"No overlaps found for {no_overlap_dither}. Defaulting to no shift")
            shift_dict[no_overlap_dither] = [0, 0]

            # Remove this from the unmatched dictionary to avoid potential weirdness
            del unmatched_dict[no_overlap_dither]

        # Step 1: Find an initial reference image, this is the first one with the largest overlap with another image
        ref_idx = np.where(all_overlap_pixels == np.nanmax(all_overlap_pixels))[0][0]
        ref_dither = dithers[ref_idx]

        log.info(f"Selected {ref_dither} as the reference dither")
        shift_dict[ref_dither] = [0, 0]

        # Step 2: Iterate until everything has a shift
        while len(shift_dict) < len(dithers):
            unmatched_dict = {}
            for dither in dithers:
                if dither not in shift_dict:
                    unmatched_dict[dither] = {}
                    unmatched_dict[dither]["max_overlap_pixels"] = 0
                    unmatched_dict[dither]["max_overlap_dither"] = None

            for matched_dither in shift_dict:
                for unmatched_dither in unmatched_dict:
                    diff = reproj_array_dict[matched_dither] - reproj_array_dict[unmatched_dither]
                    overlap_pix = diff.footprint[diff.footprint != 0]

                    n_overlap_pix = np.sum(overlap_pix)

                    # These things are symmetric, so we only need to check one way
                    if n_overlap_pix > unmatched_dict[unmatched_dither]["max_overlap_pixels"]:
                        unmatched_dict[unmatched_dither]["max_overlap_pixels"] = n_overlap_pix
                        unmatched_dict[unmatched_dither]["max_overlap_dither"] = matched_dither

            # Find the best match
            all_overlap_pixels = np.array([unmatched_dict[dither]["max_overlap_pixels"]
                                           for dither in unmatched_dict])
            all_unmatched_dithers = list(unmatched_dict.keys())
            next_idx = np.where(all_overlap_pixels == np.nanmax(all_overlap_pixels))[0][0]

            unmatched_dither = all_unmatched_dithers[next_idx]
            ref_dither = unmatched_dict[unmatched_dither]["max_overlap_dither"]

            log.info(f"Cross-correlating {unmatched_dither} with {ref_dither}")

            diff = reproj_array_dict[ref_dither] - reproj_array_dict[unmatched_dither]

            # Pull out the median, since intensities need to be closely matched
            diff_med = np.nanmedian(diff.array[diff.footprint != 0])

            # Pull things out into matched arrays
            ref_array = np.zeros(optimal_shape)
            unmatched_array = np.zeros_like(ref_array)

            ref_array[reproj_array_dict[ref_dither].view_in_original_array] += (
                reproj_array_dict[ref_dither].array)
            ref_array[ref_array == 0] = np.nan

            # Keep track of where NaNs are
            ref_nan_idx = ~np.isfinite(ref_array)
            ref_nan_idx = np.array(ref_nan_idx, dtype=int)

            # Shift the reference array if needed, keeping track roughly of where the NaNs are
            ref_yoff, ref_xoff = shift_dict[ref_dither]
            if ref_yoff != 0 or ref_xoff != 0:
                ref_array = shift.shift2d(ref_array,
                                          deltax=ref_xoff,
                                          deltay=ref_yoff,
                                          )
                ref_nan_idx = shift.shift2d(ref_nan_idx,
                                            deltax=ref_xoff,
                                            deltay=ref_yoff,
                                            )
                ref_array[ref_nan_idx > 0.99] = np.nan

            unmatched_array[reproj_array_dict[unmatched_dither].view_in_original_array] += (
                reproj_array_dict[unmatched_dither].array)
            unmatched_array[unmatched_array == 0] = np.nan

            # Finally, cut down to just the overlap area
            ref_array = ref_array[diff.imin:diff.imax, diff.jmin:diff.jmax]
            unmatched_array = unmatched_array[diff.imin:diff.imax, diff.jmin:diff.jmax]

            # And cut down the WCS to the overlap
            hdr = optimal_wcs[diff.imin:diff.imax, diff.jmin:diff.jmax].to_header()
            hdr["NAXIS1"] = ref_array.shape[1]
            hdr["NAXIS2"] = ref_array.shape[0]

            # Subtract off the median difference from the reference array
            ref_array -= diff_med

            # Set a reasonable size for the matrix from the array shapes
            num_per_dimension = np.max(unmatched_array.shape) // 2

            # Get a guess for initial shifts
            gt = align.AlignTranslationPCC(ref_array,
                                           unmatched_array,
                                           header=hdr,
                                           verbose=False,
                                           )
            init_shifts = gt.get_translation(split_image=3)

            log.info(f"Found initial shifts of [{init_shifts[1]:.3f}, {init_shifts[0]:.3f}] (pixels)")

            # Run the full optical flow
            op = align.AlignOpticalFlow(ref_array,
                                        unmatched_array,
                                        guess_translation=init_shifts,
                                        header=hdr,
                                        verbose=False,
                                        )
            op.get_iterate_translation_rotation(nruns_opticalflow=5,
                                                homography_method=TranslationTransform,
                                                num_per_dimension=num_per_dimension,
                                                oflow_test=False,
                                                )

            y_off, x_off = op.translation

            log.info(f"Found final shifts of [{x_off:.3f}, {y_off:.3f}] (pixels)")

            shift_dict[unmatched_dither] = [y_off, x_off]

            del unmatched_dict[unmatched_dither]

        # We have shifts! Now write them into the files
        for dither in dithers:

            dither_shift = shift_dict[dither]

            # Don't write anything out for 0 shifts
            if dither_shift[0] == 0 and dither_shift[1] == 0:
                continue

            # We need to swap the y sign around, not x since RA decreases left to right
            dither_shift[0] *= -1

            stacked_images = glob.glob(os.path.join(out_dir, f"{dither}*{self.in_ext}.fits"))
            for stacked_image in stacked_images:
                with fits.open(stacked_image) as hdu:
                    pixscale = get_pixscale(hdu["SCI"])
                    hdu[0].header["YSHIFT"] = dither_shift[0] * pixscale / RAD_TO_ARCSEC
                    hdu[0].header["XSHIFT"] = dither_shift[1] * pixscale / RAD_TO_ARCSEC
                    hdu.writeto(stacked_image.replace(self.in_ext, self.out_ext), overwrite=True)

        return True

    def make_stacked_images(
            self,
            dithers,
            in_dir,
            out_dir,
            procs=1,
    ):
        """Function to parallellise up making stacked dither images

        Args:
            dithers: List of dithers to go
            in_dir: Where to find files
            out_dir: Where to save stacked images to
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
                            in_dir=in_dir,
                            out_dir=out_dir,
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
            in_dir,
            out_dir,
    ):
        """Light wrapper around parallelising the stacked image

        Args:
            dither: Dither group to consider
            in_dir: Where to find files
            out_dir: Where to save stacked images to
        """

        files = glob.glob(
            os.path.join(
                in_dir,
                f"{dither}*_{self.in_ext}.fits",
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
            reproject_func=self.reproject_func,
            match_background=True,
        )
        if not success:
            return False

        return True

    def get_visit_transforms(self,
                             in_dir,
                             ):
        """Get visit transforms out of the shifted files

        Args:
            in_dir: Directory containing files with shifts
        """

        visit_transforms = {}

        log.info(f"Getting transforms")

        output_files = glob.glob(
            os.path.join(
                in_dir,
                f"*_{self.out_ext}.fits",
            )
        )

        for output_file in output_files:
            # Get matrix and (x, y) shifts from the output file, if they exist

            # If we've got _*_, this has to come from cross-correlation, so treat as a fits file
            if "_*_" in output_file:
                with fits.open(output_file) as aligned_model:
                    try:
                        translation = np.array([aligned_model[0].header["XSHIFT"], aligned_model[0].header["YSHIFT"]])
                        matrix = np.array([[1, 0], [0, 1]])
                        out_split = os.path.split(output_file)[-1]

                        # Since by default we group up dithers, don't discriminate here
                        visit = out_split.split("_")[0]

                    except KeyError:
                        continue

            # Else, we can treat as a datamodel
            else:
                with datamodels.open(output_file) as aligned_model:
                    try:
                        transform = aligned_model.meta.wcs.forward_transform["tp_affine"]
                        translation = transform.translation.value
                        matrix = transform.matrix.value

                        # Pull out a visit name. This will be different if the band is having
                        # dithers grouped or not
                        out_split = os.path.split(output_file)[-1]

                        band_type = aligned_model.meta.instrument.name.strip().lower()
                        if band_type in self.group_dithers:
                            visit = out_split.split("_")[0]
                        else:
                            visit = "_".join(out_split.split("_")[:3])

                    except IndexError:
                        continue

            xy_shift = RAD_TO_ARCSEC * translation

            if visit in visit_transforms:
                visit_transforms[visit]["shift"] = np.vstack(
                    (visit_transforms[visit]["shift"], xy_shift)
                )
                visit_transforms[visit]["matrix"] = np.dstack(
                    (visit_transforms[visit]["matrix"], matrix)
                )
            else:
                visit_transforms[visit] = {}
                visit_transforms[visit]["shift"] = copy.deepcopy(xy_shift)
                visit_transforms[visit]["matrix"] = copy.deepcopy(matrix)
            del aligned_model

        # Remove the temp directory
        shutil.rmtree(in_dir)

        # Sort the dictionary so the file is more human-readable
        visit_transforms = dict(sorted(visit_transforms.items()))

        return visit_transforms
