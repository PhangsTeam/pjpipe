import copy
import glob
import logging
import os
import shutil

import numpy as np
from astropy.table import Table, QTable
from jwst.datamodels import ModelContainer
from jwst.tweakreg import TweakRegStep
from stdatamodels.jwst import datamodels

from ..utils import get_band_type, fwhms_pix, parse_parameter_dict, recursive_setattr

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

RAD_TO_ARCSEC = 3600 * np.rad2deg(1)


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
            bands=None,
            alignment_catalogs=None,
            group_dithers=None,
            tweakreg_parameters=None,
            overwrite=False,
    ):
        """Gets a table of WCS corrections to apply to visit groups

        Experience has shown that the relative JWST guide star uncertainty is very
        small, but there are significant absolute corrections between guide stars.
        Thus, we can use the same visit as a correction for all visits, for example
        using F770W/F1000W at F2100W where tweakreg doesn't work so well.

        Here, we take some template bands and loop over with tweakreg, writing out a table
        of shifts/matrices to apply to other bands. For multiple dithers etc., will take
        an average correction

        Args:
            directory: Directory of target
            progress_dict: The progress dictionary the pipeline builds up.
                This is used to figure out what subdirectories we should
                be looking in
            target: Target to consider
            alignment_dir: Directory for alignment catalogs
            bands: List of target bands to pull corrections out for
            alignment_catalogs: Dictionary mapping targets to alignment catalogs
            group_dithers: Which band type (e.g. nircam) to group
                up dithers for and find a single correction. Defaults
                to None, which won't group up anything
            tweakreg_parameters: Dictionary of parameters to pass to tweakreg.
                Defaults to None, which will use observatory defaults
            overwrite: Whether to overwrite or not. Defaults to False
        """

        if bands is None:
            raise ValueError("Need some bands to get WCS adjustments")

        if group_dithers is None:
            group_dithers = []
        if tweakreg_parameters is None:
            tweakreg_parameters = {}
        if alignment_catalogs is None:
            alignment_catalogs = {}

        self.directory = directory
        self.progress_dict = progress_dict
        self.target = target
        self.alignment_dir = alignment_dir
        self.bands = bands
        self.alignment_catalogs = alignment_catalogs
        self.group_dithers = group_dithers
        self.tweakreg_parameters = tweakreg_parameters
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

        # Get transforms
        visit_transforms = self.get_visit_transforms()

        # Write transforms
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

    def get_visit_transforms(self):
        """Get transforms per-visit, running tweakreg and pulling out corrections"""

        in_ext = "cal"
        out_ext = "wcs_adjust"

        visit_transforms = {}

        out_dir = os.path.join(self.directory, "get_wcs_adjust")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        log.info(f"Getting transforms")

        for band_full in self.bands:
            if "bgr" in band_full:
                band = band_full.replace("_bgr", "")
            else:
                band = copy.deepcopy(band_full)

            band_type = get_band_type(band)

            # Some various failure states
            if band_full not in self.progress_dict:
                log.warning(f"No data found for {band_full}. Skipping")
                continue
            if "dir" not in self.progress_dict[band_full]:
                log.warning(f"No files found for {band_full}. Skipping")
                continue
            if not self.progress_dict[band_full]["success"]:
                log.warning(f"Previous failures found for {band_full}. Skipping")
                continue

            band_dir = copy.deepcopy(self.progress_dict[band_full]["dir"])
            if not os.path.exists(band_dir):
                log.warning(f"Directory {band_dir} does not exist")
                continue

            fwhm_pix = fwhms_pix[band]

            in_files = glob.glob(
                os.path.join(
                    band_dir,
                    f"*_{in_ext}.fits",
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
                continue

            tweakreg_config = TweakRegStep.get_config_from_reference(asn_file)
            tweakreg = TweakRegStep.from_config_section(tweakreg_config)
            tweakreg.output_dir = out_dir
            tweakreg.save_results = True
            tweakreg.suffix = out_ext
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

        output_files = glob.glob(
            os.path.join(
                out_dir,
                f"*_{out_ext}.fits",
            )
        )

        for output_file in output_files:
            # Get matrix and (x, y) shifts from the output file, if they exist
            with datamodels.open(output_file) as aligned_model:
                try:
                    transform = aligned_model.meta.wcs.forward_transform["tp_affine"]
                    matrix = transform.matrix.value
                    xy_shift = RAD_TO_ARCSEC * transform.translation.value

                    # Pull out a visit name. This will be different if the band is having
                    # dithers grouped or not
                    out_split = os.path.split(output_file)[-1]

                    band_type = aligned_model.meta.instrument.name.strip().lower()
                    if band_type in self.group_dithers:
                        visit = out_split.split("_")[0]
                    else:
                        visit = "_".join(out_split.split("_")[:3])

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

                except IndexError:
                    pass
            del aligned_model

        # Remove the temp directory
        shutil.rmtree(out_dir)

        # Sort the dictionary so the file is more human-readable
        visit_transforms = dict(sorted(visit_transforms.items()))

        return visit_transforms
