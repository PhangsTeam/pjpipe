import copy
import gc
import glob
import json
import logging
import os
import shutil
import time

import asdf
import numpy as np
import jwst
from jwst.datamodels import ModelContainer
from jwst.resample import ResampleStep
from jwst.resample.resample import compute_image_pixel_area
from stcal.outlier_detection.utils import gwcs_blot

# FIXME For newer JWST versions, import ModelLibrary
try:
    from jwst.datamodels import ModelLibrary
except ImportError:
    class ModelLibrary:
        pass

from jwst.pipeline import calwebb_image3
from jwst.skymatch import SkyMatchStep
from jwst.tweakreg import TweakRegStep
from stdatamodels.jwst import datamodels

from ..utils import (
    get_band_type,
    get_band_ext,
    get_obs_table,
    parse_parameter_dict,
    attribute_setter,
    recursive_setattr,
    fwhms_pix,
    save_file,
    get_short_band_name,
)

log = logging.getLogger(__name__)

BGR_CHECK_TYPES = [
    "parallel_off",
    "check_in_name",
    "filename",
]


class Lv3Step:
    def __init__(
            self,
            target,
            band,
            in_dir,
            out_dir,
            dr_version,
            is_bgr,
            step_ext,
            procs,
            tweakreg_degroup_nircam_modules=False,
            tweakreg_degroup_nircam_short_chips=False,
            tweakreg_group_dithers=None,
            tweakreg_degroup_dithers=None,
            skymatch_group_dithers=None,
            skymatch_degroup_dithers=None,
            bgr_check_type="parallel_off",
            bgr_background_name="off",
            process_bgr_like_science=False,
            jwst_parameters=None,
            do_drizzle=None,
            do_blot=False,
            blot_ref_index=0,
            blot_fillval=np.nan,
            overwrite=False,
    ):
        """Wrapper around the level 3 JWST pipeline

        Args:
            target: Target to consider
            band: Band to consider
            in_dir: Input directory
            out_dir: Output directory
            dr_version: Data processing version
            is_bgr: Whether we're processing background observations or not
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel
            tweakreg_degroup_nircam_modules: Whether to degroup NIRCam A and B
                modules. Currently, the WCS is inconsistent between the two,
                so should probably be set to True if you see "ghosting" in the final
                mosaic. Defaults to False
            tweakreg_degroup_nircam_short_chips: Whether to degroup NIRCam short 1/2/3/4
                chips. There may be some shifts between these, so should ideally find a shift
                for each chip. Defaults to False
            tweakreg_group_dithers: List of 'miri',
                'nircam_long', 'nircam_short' of whether to group
                up dithers for tweakreg. Defaults to None, which will
                keep at default
            tweakreg_degroup_dithers: List of 'miri', 'nircam_long',
                'nircam_short' of whether to degroup dithers for
                tweakreg. Defaults to None, which will keep at
                default.
            skymatch_group_dithers: List of 'miri', 'nircam_long',
                'nircam_short' of whether to group up dithers for
                skymatch. Defaults to None, which will keep at
                default
            skymatch_degroup_dithers: List of 'miri', 'nircam_long',
                'nircam_short' of whether to degroup dithers for
                skymatch. Defaults to None, which will keep at
                default.
            bgr_check_type: Method to check if obs is science
                or background. Options are given by BGR_CHECK_TYPES.
                Defaults to 'parallel_off'
            bgr_background_name: If `bgr_check_type` is 'check_in_name'
                or 'filename', this is the string to match
            process_bgr_like_science: If True, will process background
                images as if they are science images. Defaults to False
            jwst_parameters: Parameter dictionary to pass to
                the level 2 pipeline. Defaults to None,
                which will run the observatory defaults
            do_drizzle: If True, drizzle individual frames
                to the i2d mosaic WCS after the main
                pipeline run. Note: creates a lot of files.
                If not set (None) and do_blot is True,
                do_drizzle is automatically enabled. If
                explicitly set to False while do_blot is
                True, a ValueError is raised. Defaults to
                None.
            do_blot: If True, blot each resampled exposure
                back to a reference detector frame (requires
                drizzling). Defaults to False.
            blot_ref_index: Index into the alphabetically sorted CRF file list of
                the reference exposure whose detector frame
                is used for blotting. Defaults to 0. 
            blot_fillval: Fill value for pixels outside the
                blotted footprint. Defaults to np.nan
            overwrite: Whether to overwrite or not. Defaults
                to False
        """
        if jwst_parameters is None:
            jwst_parameters = {}

        if tweakreg_group_dithers is None:
            tweakreg_group_dithers = []
        if tweakreg_degroup_dithers is None:
            tweakreg_degroup_dithers = []
        if skymatch_group_dithers is None:
            skymatch_group_dithers = []
        if skymatch_degroup_dithers is None:
            skymatch_degroup_dithers = []

        if is_bgr:
            bgr_ext = "_bgr"
        else:
            bgr_ext = ""

        self.target = target
        self.band = band
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.dr_version = dr_version
        self.is_bgr = is_bgr
        self.bgr_ext = bgr_ext
        self.step_ext = step_ext
        self.procs = procs
        self.tweakreg_degroup_nircam_modules = tweakreg_degroup_nircam_modules
        self.tweakreg_degroup_nircam_short_chips = tweakreg_degroup_nircam_short_chips
        self.tweakreg_group_dithers = tweakreg_group_dithers
        self.tweakreg_degroup_dithers = tweakreg_degroup_dithers
        self.skymatch_group_dithers = skymatch_group_dithers
        self.skymatch_degroup_dithers = skymatch_degroup_dithers
        self.bgr_check_type = bgr_check_type
        self.bgr_background_name = bgr_background_name
        self.process_bgr_like_science = process_bgr_like_science
        self.jwst_parameters = jwst_parameters
        if do_drizzle is False and do_blot:
            raise ValueError(
                "do_drizzle was explicitly set to False, but do_blot "
                "was set to True. Either set (do_drizzle, do_blot) to "
                "(True, True) or (True, False), or (None, True)."
            )
        if do_drizzle is None:
            do_drizzle = bool(do_blot)
        self.do_drizzle = do_drizzle
        self.do_blot = do_blot
        self.blot_ref_index = blot_ref_index
        self.blot_fillval = blot_fillval
        self.overwrite = overwrite

        self.band_type = get_band_type(self.band)
        self.band_ext = get_band_ext(self.band)

    def do_step(self):
        """Run the level 3 pipeline"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)
            os.system(f"rm -rf {os.path.join(self.in_dir, '*.json')}")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "lv3_step_complete.txt",
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

        asn_file = self.create_asn_file(
            files=files,
        )

        success = self.run_step(
            asn_file,
        )

        # If not everything has succeeded, then return a warning
        if not success:
            log.warning("Failures detected in level 3 pipeline")
            return False

        # Propagate PJPipe metadata through the output files, this is the
        # lv3 mosaic and the crf files
        all_files = []
        files = glob.glob(os.path.join(self.out_dir,
                                       "*_i2d.fits",
                                       )
                          )
        all_files.extend(files)
        files = glob.glob(os.path.join(self.out_dir,
                                       "*_crf.fits",
                                       )
                          )
        all_files.extend(files)
        self.propagate_metadata(all_files)

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def create_asn_file(
            self,
            files,
    ):
        """Setup asn lv3 file"""

        log.info("Building asn file")

        check_bgr = True
        band_short = get_short_band_name(self.band)

        # If we have NIRCam/NIRISS operating with parallel offs, switch off background checking
        # else everything will be flagged as backgrounds
        if self.band_type in ["nircam", "niriss"] and self.bgr_check_type == "parallel_off":
            check_bgr = False

        asn_lv3_filename = os.path.join(
            self.in_dir,
            f"asn_lv3_{self.band}.json",
        )

        files.sort()

        tab = get_obs_table(
            files=files,
            check_bgr=check_bgr,
            check_type=self.bgr_check_type,
            background_name=self.bgr_background_name,
        )

        json_content = {
            "asn_type": "None",
            "asn_rule": "DMS_Level3_Base",
            "version_id": time.strftime("%Y%m%dt%H%M%S"),
            "code_version": jwst.__version__,
            "degraded_status": "No known degraded exposures in association.",
            "program": tab["Program"][0],
            "constraints": "No constraints",
            "asn_id": f"o{tab['Obs_ID'][0]}",
            "asn_pool": "none",
            "products": [
                {
                    "name": f"{self.target.lower()}_{self.band_type}_lv3_{band_short.lower()}{self.bgr_ext}",
                    "members": [],
                }
            ],
        }

        # If we're only processing background, flip the switch
        if self.is_bgr:
            tab["Type"] = "sci"

        # If we're processing background like science, flip the switch
        if self.process_bgr_like_science:
            tab["Type"] = "sci"

        # Only take things flagged as science
        tab = tab[tab["Type"] == "sci"]

        for row in tab:
            json_content["products"][-1]["members"].append(
                {"expname": row["File"], "exptype": "science", "exposerr": "null"}
            )

        with open(asn_lv3_filename, "w") as f:
            json.dump(json_content, f)

        return asn_lv3_filename

    def run_step(
            self,
            asn_file,
    ):
        """Run the level 3 step

        Args:
            asn_file: Path to association JSON file
        """

        log.info("Running level 3 pipeline")

        band_type, short_long = get_band_type(self.band, short_long_nircam=True)
        band_short = get_short_band_name(self.band)

        # FWHM should be set per-band for both tweakreg and source catalogue
        fwhm_pix = fwhms_pix[band_short]

        # Set up to run lv3 pipeline

        config = calwebb_image3.Image3Pipeline.get_config_from_reference(asn_file)
        im3 = calwebb_image3.Image3Pipeline.from_config_section(config)
        im3.output_dir = self.out_dir

        im3.tweakreg.kernel_fwhm = fwhm_pix  # * 2
        im3.source_catalog.kernel_fwhm = fwhm_pix  # * 2

        im3 = attribute_setter(
            im3,
            parameters=self.jwst_parameters,
            band=self.band,
            target=self.target,
        )

        # Load the asn file in, so we have access to everything we need later
        asn_file = ModelContainer(asn_file)

        # Run the tweakreg step
        config = TweakRegStep.get_config_from_reference(asn_file)
        tweakreg = TweakRegStep.from_config_section(config)
        tweakreg.output_dir = self.out_dir
        tweakreg.save_results = False
        tweakreg.kernel_fwhm = fwhm_pix  # * 2

        try:
            tweakreg_params = self.jwst_parameters["tweakreg"]
        except KeyError:
            tweakreg_params = {}

        for tweakreg_key in tweakreg_params:
            value = parse_parameter_dict(
                parameters=tweakreg_params,
                key=tweakreg_key,
                band=self.band,
                target=self.target,
            )

            if value == "VAL_NOT_FOUND":
                continue

            recursive_setattr(tweakreg, tweakreg_key, value)

        # Keep track of exposure numbers and group IDs in case we change them
        meta_params = {}
        for model in asn_file._models:
            model_name = model.meta.filename

            if hasattr(model.meta.observation, "exposure_number"):
                exp_no = copy.deepcopy(model.meta.observation.exposure_number)
            else:
                exp_no = ""

            if hasattr(model.meta, "group_id"):
                group_id = copy.deepcopy(model.meta.group_id)
            else:
                group_id = ""

            meta_params[model_name] = [exp_no,
                                       group_id,
                                       ]

        # Group up the dithers
        if short_long in self.tweakreg_group_dithers:
            for model in asn_file._models:
                model.meta.observation.exposure_number = "1"
                model.meta.group_id = ""

        # Or degroup the dithers
        elif short_long in self.tweakreg_degroup_dithers:
            for i, model in enumerate(asn_file._models):
                model.meta.observation.exposure_number = str(i)
                model.meta.group_id = ""

        # If needed, degroup the NIRCam modules. Do this by adding a large
        # number to the exposure number
        if (
                band_type == "nircam"
                and self.tweakreg_degroup_nircam_modules
        ):
            for i, model in enumerate(asn_file._models):
                module = model.meta.instrument.module.strip().lower()

                exp_no = int(model.meta.observation.exposure_number)
                if module == "a":
                    exp_add = 99
                elif module == "b":
                    exp_add = 100
                else:
                    raise ValueError("Expecting module to either be A or B")

                model.meta.observation.exposure_number = str(exp_no + exp_add)
                model.meta.group_id = ""

        # Degroup the 1/2/3/4 NIRCam shorts, if requested
        if (
                band_type == "nircam"
                and self.tweakreg_degroup_nircam_short_chips
        ):
            for i, model in enumerate(asn_file._models):
                detector = model.meta.instrument.detector.strip().lower()

                exp_no = int(model.meta.observation.exposure_number)

                # Include information from the particular chip if we're in short
                # mode (i.e. there's a 1-4 in the detector name), and keep
                # track of this to modify the group ID
                exp_add = 0
                if "1" in detector:
                    exp_add += 49
                elif "2" in detector:
                    exp_add += 50
                elif "3" in detector:
                    exp_add += 51
                elif "4" in detector:
                    exp_add += 52

                model.meta.observation.exposure_number = str(exp_no + exp_add)
                model.meta.group_id = ""

        asn_file = tweakreg.run(asn_file)

        # If the asn file is a ModelLibrary, we need to force the name back in
        use_model_library = False

        if isinstance(asn_file, ModelLibrary):
            use_model_library = True

            # If there's no final name here, add it now
            if "name" not in asn_file._asn["products"][0]:
                name = f"{self.target.lower()}_{self.band_type}_lv3_{band_short.lower()}{self.bgr_ext}"
                asn_file._asn["products"][0]["name"] = copy.deepcopy(name)

        del tweakreg
        gc.collect()

        # Make sure we skip tweakreg since we've already done it
        im3.tweakreg.skip = True

        if use_model_library:
            models = asn_file._loaded_models
        else:
            models = asn_file._models

        # Reset if we're degrouping NIRCam modules
        if (
                band_type == "nircam"
                and self.tweakreg_degroup_nircam_modules
        ):
            if use_model_library:
                for i in models:
                    model_name = models[i].meta.filename
                    models[i].meta.observation.exposure_number = meta_params[model_name][0]
                    models[i].meta.group_id = meta_params[model_name][1]

            else:
                for i, model in enumerate(models):
                    model_name = model.meta.filename
                    model.meta.observation.exposure_number = meta_params[model_name][0]
                    model.meta.group_id = meta_params[model_name][1]

        # Remove the chip info if we're degrouping the NIRCam short chips
        if (
                band_type == "nircam"
                and self.tweakreg_degroup_nircam_short_chips
        ):

            if use_model_library:
                for i in models:
                    model_name = models[i].meta.filename
                    models[i].meta.observation.exposure_number = meta_params[model_name][0]
                    models[i].meta.group_id = meta_params[model_name][1]

            else:
                for i, model in enumerate(models):
                    model_name = model.meta.filename
                    model.meta.observation.exposure_number = meta_params[model_name][0]
                    model.meta.group_id = meta_params[model_name][1]

        # Set meta parameters back to original values for group/degrouping of dithers
        if (
                short_long in self.tweakreg_group_dithers
                or short_long in self.tweakreg_degroup_dithers
        ):

            if use_model_library:
                for i in models:
                    model_name = models[i].meta.filename
                    models[i].meta.observation.exposure_number = meta_params[model_name][0]
                    models[i].meta.group_id = meta_params[model_name][1]

            else:
                for i, model in enumerate(models):
                    model_name = model.meta.filename
                    model.meta.observation.exposure_number = meta_params[model_name][0]
                    model.meta.group_id = meta_params[model_name][1]

        # Run the skymatch step with custom hacks if required
        config = SkyMatchStep.get_config_from_reference(asn_file)
        skymatch = SkyMatchStep.from_config_section(config)
        skymatch.output_dir = self.out_dir
        skymatch.save_results = False

        try:
            skymatch_params = self.jwst_parameters["skymatch"]
        except KeyError:
            skymatch_params = {}

        for skymatch_key in skymatch_params:
            value = parse_parameter_dict(
                parameters=skymatch_params,
                key=skymatch_key,
                band=self.band,
                target=self.target,
            )

            if value == "VAL_NOT_FOUND":
                continue

            recursive_setattr(skymatch, skymatch_key, value)

        # Group or degroup for skymatching
        if short_long in self.skymatch_group_dithers:

            if use_model_library:
                for i in models:
                    models[i].meta.observation.exposure_number = "1"
                    models[i].meta.group_id = ""

            else:
                for model in models:
                    model.meta.observation.exposure_number = "1"
                    model.meta.group_id = ""

        elif short_long in self.skymatch_degroup_dithers:

            if use_model_library:
                for i in models:
                    models[i].meta.observation.exposure_number = str(i)
                    models[i].meta.group_id = ""

            else:
                for i, model in enumerate(models):
                    model.meta.observation.exposure_number = str(i)
                    model.meta.group_id = ""

        asn_file = skymatch.run(asn_file)

        del skymatch
        gc.collect()

        use_model_library = False
        if isinstance(asn_file, ModelLibrary):
            use_model_library = True

            # If there's no final name here, add it now
            if "name" not in asn_file._asn["products"][0]:
                name = f"{self.target.lower()}_{self.band_type}_lv3_{band_short.lower()}{self.bgr_ext}"
                asn_file._asn["products"][0]["name"] = copy.deepcopy(name)

        if use_model_library:
            models = asn_file._loaded_models
        else:
            models = asn_file._models

        # Set meta parameters back to original values to avoid potential weirdness later
        if (
                short_long in self.skymatch_group_dithers
                or short_long in self.skymatch_degroup_dithers
        ):

            if use_model_library:
                for i in models:
                    model_name = models[i].meta.filename
                    models[i].meta.observation.exposure_number = meta_params[model_name][0]
                    models[i].meta.group_id = meta_params[model_name][1]

            else:
                for i, model in enumerate(models):
                    model_name = model.meta.filename
                    model.meta.observation.exposure_number = meta_params[model_name][0]
                    model.meta.group_id = meta_params[model_name][1]

        im3.skymatch.skip = True

        # Run the rest of the level 3 pipeline

        if use_model_library:
            # Re-instantiate the ModelLibrary, to wipe out any weirdness we might have performed
            # along the way
            asn_file = ModelContainer([models[m] for m in models])
            asn_file = ModelLibrary(asn_file, on_disk=False)
            if "name" not in asn_file._asn["products"][0]:
                name = f"{self.target.lower()}_{self.band_type}_lv3_{band_short.lower()}{self.bgr_ext}"
                asn_file._asn["products"][0]["name"] = copy.deepcopy(name)

        im3.run(asn_file)

        del im3
        del asn_file
        gc.collect()

        # Drizzle individual frames to the common mosaic WCS
        if self.do_drizzle:
            i2d_files = glob.glob(os.path.join(self.out_dir, "*_i2d.fits"))
            crf_files = sorted(glob.glob(os.path.join(self.out_dir, "*_crf.fits")))

            if i2d_files and crf_files:
                ref_wcs_file = os.path.join(self.out_dir, "resample_refwcs.asdf")
                with datamodels.open(i2d_files[0]) as i2d_model:
                    asdf.AsdfFile({
                        "wcs": i2d_model.meta.wcs,
                        "array_shape": i2d_model.data.shape,
                    }).write_to(ref_wcs_file)

                resample_single = ResampleStep()
                resample_single.single = True
                resample_single.output_use_model = True
                resample_single.save_results = True
                resample_single.output_dir = self.out_dir
                resample_single.output_wcs = ref_wcs_file

                try:
                    resample_params = self.jwst_parameters["resample"]
                except KeyError:
                    resample_params = {}

                wcs_keys = {
                    "output_wcs", "crval", "crpix", "rotation",
                    "pixel_scale", "pixel_scale_ratio", "output_shape",
                }
                for resample_key in resample_params:
                    if resample_key in wcs_keys:
                        continue
                    value = parse_parameter_dict(
                        parameters=resample_params,
                        key=resample_key,
                        band=self.band,
                        target=self.target,
                    )
                    if value == "VAL_NOT_FOUND":
                        continue
                    recursive_setattr(resample_single, resample_key, value)

                resample_single.run(crf_files)

                del resample_single
                gc.collect()

                # Notes:
                #   - ResampleImage.resample_many_to_many() adds _outlier_resamplestep.fits
                #   to the file names, but it's not an outlier image, so let's rename it.
                #   - We'd have to edit ResampleStep for a proper fix.
                #   - Can't have suffix ending in _i2d.fits, messes with anchoring pattern matching.
                for f in glob.glob(os.path.join(self.out_dir, "*_outlier_resamplestep.fits")):
                    os.rename(f, f.replace("_outlier_resamplestep.fits", "_i2d_single.fits"))

                if self.do_blot:
                    self._blot_to_detector_frame(
                        crf_files=crf_files,
                        ref_index=self.blot_ref_index,
                    )
            else:
                log.warning("do_drizzle is set but no i2d/crf files found")

        return True

    def _blot_to_detector_frame(
            self,
            crf_files,
            ref_index=0,
    ):
        """Blot resampled single-exposure images back to a detector frame.

        Each ``*_i2d_single.fits`` file (one per exposure, on the common
        i2d WCS grid) is blotted onto the detector frame of the chosen
        reference exposure.  The output files are written as
        ``*_blot_det.fits`` alongside the other lv3 products.

        Args:
            crf_files: Sorted list of CRF file paths (one per exposure).
                Used to obtain the detector-frame GWCS and shape.
            ref_index: Index into *crf_files* of the reference exposure
                whose detector frame will be adopted.  Defaults to 0
                (the first exposure).
        """

        single_files = sorted(
            glob.glob(os.path.join(self.out_dir, "*_i2d_single.fits"))
        )
        if not single_files:
            log.warning("_blot_to_detector_frame: no *_i2d_single.fits files found")
            return

        if ref_index >= len(crf_files):
            log.warning(
                "_blot_to_detector_frame: ref_index %d out of range (%d exposures)",
                ref_index, len(crf_files),
            )
            return

        log.info(
            "Blotting %d resampled exposures to detector frame of %s",
            len(single_files), os.path.basename(crf_files[ref_index]),
        )

        with datamodels.open(crf_files[ref_index]) as ref_model:
            blot_wcs = ref_model.meta.wcs
            blot_shape = ref_model.data.shape

            ref_pixflux_area = ref_model.meta.photometry.pixelarea_steradians
            blot_wcs.array_shape = blot_shape
            ref_pixel_area = compute_image_pixel_area(blot_wcs)
            pix_ratio = np.sqrt(ref_pixflux_area / ref_pixel_area)

            for single_file in single_files:
                with datamodels.open(single_file) as single_model:
                    blotted = gwcs_blot(
                        median_data=single_model.data.astype(np.float32),
                        median_wcs=single_model.meta.wcs,
                        blot_shape=blot_shape,
                        blot_wcs=blot_wcs,
                        pix_ratio=pix_ratio,
                        fillval=self.blot_fillval,
                    )

                out_name = single_file.replace("_i2d_single.fits", "_blot_det.fits")
                blot_model = datamodels.ImageModel(data=blotted)
                blot_model.update(ref_model)
                blot_model.meta.wcs = copy.deepcopy(blot_wcs)
                save_file(blot_model, out_name=out_name, dr_version=self.dr_version)
                blot_model.close()

                log.info(
                    "  %s -> %s",
                    os.path.basename(single_file),
                    os.path.basename(out_name),
                )

        gc.collect()

    def propagate_metadata(self,
                           files):
        """Propagate metadata through to the output files

        Args:
            files: List of files to loop over
        """

        for out_name in files:
            with datamodels.open(out_name) as im:
                save_file(im, out_name=out_name, dr_version=self.dr_version)

            del im

        return True
