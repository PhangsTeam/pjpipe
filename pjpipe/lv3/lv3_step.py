import copy
import gc
import glob
import json
import logging
import os
import shutil
import time

import jwst
from jwst.datamodels import ModelContainer

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
)

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

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

        # If we have NIRCam operating with parallel offs, switch off background checking
        # else everything will be flagged as backgrounds
        if self.band_type == "nircam" and self.bgr_check_type == "parallel_off":
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
                    "name": f"{self.target.lower()}_{self.band_type}_lv3_{self.band.lower()}{self.bgr_ext}",
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

        # FWHM should be set per-band for both tweakreg and source catalogue
        fwhm_pix = fwhms_pix[self.band]

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
            meta_params[model_name] = [model.meta.observation.exposure_number,
                                       model.meta.group_id,
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

        # If needed, degroup the NIRCam modules. Do this by changing the
        # first letter of the filename to the module, and adding a large
        # number to the exposure number to degroup them.
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

                model_name = list(copy.deepcopy(model.meta.filename))
                model_name[0] = module
                model_name = "".join(model_name)
                model.meta.filename = copy.deepcopy(model_name)

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
                # track of this to modify the name later
                exp_add = 0
                if "1" in detector:
                    exp_add += 49
                    det = "1"
                elif "2" in detector:
                    exp_add += 50
                    det = "2"
                elif "3" in detector:
                    exp_add += 51
                    det = "3"
                elif "4" in detector:
                    exp_add += 52
                    det = "4"
                else:
                    det = "l"

                model.meta.observation.exposure_number = str(exp_no + exp_add)
                model.meta.group_id = ""

                model_name = list(copy.deepcopy(model.meta.filename))
                model_name.insert(1, det)
                model_name = "".join(model_name)
                model.meta.filename = copy.deepcopy(model_name)

        asn_file = tweakreg.run(asn_file)

        # If the asn file is a ModelLibrary, we need to force the name back in
        use_model_library = False

        if isinstance(asn_file, ModelLibrary):
            use_model_library = True

            # If there's no final name here, add it now
            if "name" not in asn_file._asn["products"][0]:
                name = f"{self.target.lower()}_{self.band_type}_lv3_{self.band.lower()}{self.bgr_ext}"
                asn_file._asn["products"][0]["name"] = copy.deepcopy(name)

        del tweakreg
        gc.collect()

        # Make sure we skip tweakreg since we've already done it
        im3.tweakreg.skip = True

        if use_model_library:
            models = asn_file._loaded_models
        else:
            models = asn_file._models

        # Set the name back to "jw" at the start if we're degrouping NIRCam modules
        if (
                band_type == "nircam"
                and self.tweakreg_degroup_nircam_modules
        ):
            if use_model_library:
                for i in models:
                    model_name = list(copy.deepcopy(models[i].meta.filename))
                    model_name[0] = "j"
                    model_name = "".join(model_name)
                    models[i].meta.filename = copy.deepcopy(model_name)
                    models[i].meta.observation.exposure_number = meta_params[model_name][0]
                    models[i].meta.group_id = meta_params[model_name][1]

            else:
                for i, model in enumerate(models):
                    model_name = list(copy.deepcopy(model.meta.filename))
                    model_name[0] = "j"
                    model_name = "".join(model_name)
                    model.meta.filename = copy.deepcopy(model_name)
                    model.meta.observation.exposure_number = meta_params[model_name][0]
                    model.meta.group_id = meta_params[model_name][1]

        # Remove the chip info if we're degrouping the NIRCam short chips
        if (
                band_type == "nircam"
                and self.tweakreg_degroup_nircam_short_chips
        ):

            if use_model_library:
                for i in models:
                    model_name = list(copy.deepcopy(models[i].meta.filename))
                    model_name.pop(1)
                    model_name = "".join(model_name)
                    models[i].meta.filename = copy.deepcopy(model_name)
                    models[i].meta.observation.exposure_number = meta_params[model_name][0]
                    models[i].meta.group_id = meta_params[model_name][1]

            else:
                for i, model in enumerate(models):
                    model_name = list(copy.deepcopy(model.meta.filename))
                    model_name.pop(1)
                    model_name = "".join(model_name)
                    model.meta.filename = copy.deepcopy(model_name)
                    model.meta.observation.exposure_number = meta_params[model_name][0]
                    model.meta.group_id = meta_params[model_name][1]

        # Set meta parameters back to original values to avoid potential weirdness later
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
                name = f"{self.target.lower()}_{self.band_type}_lv3_{self.band.lower()}{self.bgr_ext}"
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
        im3.run(asn_file)

        del im3
        del asn_file
        gc.collect()

        return True

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
