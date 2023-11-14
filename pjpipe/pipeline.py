import copy
import logging
import os
import shutil
import multiprocessing as mp

from .apply_wcs_adjust import ApplyWCSAdjustStep
from .astrometric_align import AstrometricAlignStep
from .astrometric_catalog import AstrometricCatalogStep
from .download import DownloadStep
from .gaia_query import GaiaQueryStep
from .level_match import LevelMatchStep
from .lv1 import Lv1Step
from .lv2 import Lv2Step
from .lv3 import Lv3Step
from .lyot_mask import LyotMaskStep
from .lyot_separate import LyotSeparateStep
from .mosaic_individual_fields import MosaicIndividualFieldsStep
from .move_raw_obs import MoveRawObsStep
from .multi_tile_destripe import MultiTileDestripeStep
from .single_tile_destripe import SingleTileDestripeStep
from .get_wcs_adjust import GetWCSAdjustStep
from .anchoring import AnchoringStep
from .psf_matching import PSFMatchingStep
from .psf_model import PSFModelStep
from .release import ReleaseStep
from .regress_against_previous import RegressAgainstPreviousStep
from .utils import *

# All possible steps
ALLOWED_STEPS = [
    "download",
    "gaia_query",
    "lv1",
    "lv2",
    "single_tile_destripe",
    "get_wcs_adjust",
    "apply_wcs_adjust",
    "lyot_mask",
    "lyot_separate",
    "multi_tile_destripe",
    "level_match",
    "psf_model",
    "lv3",
    "astrometric_catalog",
    "astrometric_align",
    "mosaic_individual_fields",
    "anchoring",
    "psf_matching",
    "release",
    "regress_against_previous",
]

# Steps that don't operate per-band
COMBINED_BAND_STEPS = [
    "download",
    "gaia_query",
    "get_wcs_adjust",
    "anchoring",
    "release",
    "regress_against_previous",
]

# .fits extensions and input/output
# directories
IN_STEP_EXTS = {
    "download": None,
    "lv1": "uncal",
    "lv2": "rate",
    "astrometric_catalog": "i2d",
    "astrometric_align": "i2d",
    "anchoring": "i2d_align",
    "psf_matching": "i2d_anchor",
    "release": None,
}

IN_BAND_DIRS = {
    "lv1": "uncal",
    "lv2": "lv1",
    "astrometric_catalog": "lv3",
    "astrometric_align": "lv3",
    "anchoring": "lv3",
    "psf_matching": "lv3",
    "release": "lv3",
}

OUT_BAND_DIRS = {
    "astrometric_catalog": "lv3",
    "astrometric_align": "lv3",
    "anchoring": "anchoring",
    "release": None,
}

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class PJPipeline:
    def __init__(
        self,
        config_file=None,
        local_file=None,
    ):
        """Overall wrapper for pjpipe.

        Args:
            config_file: Either string to location of config.toml file,
                or dict of preloaded toml
            local_file: Either string to location of local.toml file,
                or dict of preloaded toml
        """

        if config_file is None:
            raise ValueError("config_file should be defined")
        elif isinstance(config_file, str):
            config = load_toml(config_file)
        elif isinstance(config_file, dict):
            config = copy.deepcopy(config_file)
        else:
            raise ValueError("config_file should be one of str, dict")

        if local_file is None:
            raise ValueError("local_file should be defined")
        elif isinstance(local_file, str):
            local = load_toml(local_file)
        elif isinstance(local_file, dict):
            local = copy.deepcopy(local_file)
        else:
            raise ValueError("local_file should be one of str, dict")

        if "webb_psf_data" in local:
            os.environ["WEBBPSF_PATH"] = local["webb_psf_data"]

        if "WEBBPSF_PATH" not in os.environ:
            log.warning(
                "WEBBPSF_PATH not set. If trying to PSF model, this will cause an error"
            )

        log.info("Starting PHANGS-JWST pipeline")

        if "crds_context" in local:
            crds_context = local["crds_context"]
            os.environ['CRDS_CONTEXT'] = copy.deepcopy(crds_context)
        else:
            crds_context = "Default"

        # Pull in needed values from the configs
        self.targets = config["targets"]
        self.bands = config["bands"]
        self.steps = config["steps"]
        self.version = config["version"]
        self.parameters = config["parameters"]
        self.raw_dir = local["raw_dir"]
        self.reprocess_dir = os.path.join(
            local["reprocess_dir"],
            self.version,
        )
        self.release_dir = os.path.join(
            local["reprocess_dir"],
            "release",
            self.version,
        )
        self.alignment_dir = local["alignment_dir"]

        if "kernel_dir" in local:
            self.kernel_dir = local["kernel_dir"]
        else:
            self.kernel_dir = None

        if "anchor_ref_dir" in local:
            self.anchor_ref_dir = local["anchor_ref_dir"]
        else:
            self.anchor_ref_dir = None

        if "processors" in local:
            procs = local["processors"]
        else:
            procs = mp.cpu_count()
        self.procs = procs

        # Log the environment variables that should be set
        log.info(f"Using CRDS_SERVER_URL: {os.environ['CRDS_SERVER_URL']}")
        log.info(f"Using CRDS_PATH: {os.environ['CRDS_PATH']}")
        log.info(f"Using CRDS_CONTEXT: {crds_context}")
        log.info(f"Using {self.procs} processes")

        # Log targets/band/steps out
        log.info("Found targets:")
        for target in self.targets:
            log.info(f"-> {target}")
        log.info(f"Found bands:")
        for band in self.bands:
            log.info(f"-> {band}")
        log.info(f"Found steps:")
        for step in self.steps:
            log.info(f"-> {step}")

    def do_pipeline(self):
        progress_dict = {}

        for target in self.targets:
            progress_dict[target] = {}

            log.info(f"Beginning reprocessing: {target}")

            for step_full in self.steps:
                # Parse out any potential instrument/observing type specific steps
                step = None
                step_instrument = None
                step_science_type = None

                if "." in step_full:
                    step_split = step_full.split(".")

                    # First search for sci/bgr
                    if "bgr" in step_split:
                        step_science_type = "bgr"
                        step_split.remove(step_science_type)
                    if "sci" in step_split:
                        step_science_type = "sci"
                        step_split.remove(step_science_type)

                    step, step_instrument = copy.deepcopy(step_split)
                else:
                    step = copy.deepcopy(step_full)

                if step not in ALLOWED_STEPS:
                    raise ValueError(
                        f"step should be one of {ALLOWED_STEPS}, not {step}"
                    )

                if step in self.parameters:
                    step_parameters = self.parameters[step]
                else:
                    step_parameters = {}

                # Get a target directory
                target_dir = os.path.join(
                    self.reprocess_dir,
                    target,
                )
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)

                # Get the default in/out .fits extension for this step
                if step in IN_STEP_EXTS:
                    in_step_ext = IN_STEP_EXTS[step]
                else:
                    in_step_ext = "cal"

                # Some steps operate on all bands, distinguish that here
                if step in COMBINED_BAND_STEPS:
                    log.info(f"Beginning {step}")

                    # Download
                    if step == "download":
                        download_dir = os.path.join(
                            self.raw_dir,
                            target,
                        )

                        download = DownloadStep(
                            target=target,
                            download_dir=download_dir,
                            procs=self.procs,
                            **step_parameters,
                        )
                        step_result = download.do_step()

                    elif step == "gaia_query":
                        gaia_query = GaiaQueryStep(
                            target=target,
                            out_dir=self.alignment_dir,
                            **step_parameters,
                        )
                        step_result = gaia_query.do_step()

                    elif step == "get_wcs_adjust":
                        get_wcs_adjust = GetWCSAdjustStep(
                            directory=target_dir,
                            progress_dict=progress_dict[target],
                            target=target,
                            alignment_dir=self.alignment_dir,
                            **step_parameters,
                        )
                        step_result = get_wcs_adjust.do_step()

                    # anchoring is in the part operating for all bands because
                    # we need more control on the sequence (reference nircam and miri bands first)
                    elif step == "anchoring":

                        in_subdir = IN_BAND_DIRS[step]
                        out_subdir = OUT_BAND_DIRS[step]

                        anchoring = AnchoringStep(
                            target=target,
                            bands=self.bands,
                            in_dir=target_dir,
                            in_subdir=in_subdir,
                            out_subdir=out_subdir,
                            ref_dir=self.anchor_ref_dir,
                            kernel_dir=self.kernel_dir,
                            in_step_ext=in_step_ext,
                            out_step_ext='i2d_anchor',
                            procs=self.procs,
                            **step_parameters,
                        )
                        step_result = anchoring.do_step()

                    elif step == "release":
                        release = ReleaseStep(
                            in_dir=target_dir,
                            out_dir=self.release_dir,
                            target=target,
                            bands=self.bands,
                            **step_parameters,
                        )
                        step_result = release.do_step()

                    elif step == "regress_against_previous":
                        regress = RegressAgainstPreviousStep(
                            target=target,
                            in_dir=self.release_dir,
                            curr_version=self.version,
                            **step_parameters,
                        )
                        step_result = regress.do_step()

                    else:
                        raise ValueError(
                            f"step should be one of {ALLOWED_STEPS}, not {step}"
                        )

                    # If we're not successful here, log a warning and delete the whole target folder
                    if not step_result:
                        log.warning(
                            f"Failures detected for {target}. "
                            f"Will remove target directory and continue."
                        )
                        shutil.rmtree(target_dir)

                    log.info(f"Completed {step}")

                else:
                    for band_full in self.bands:
                        if "bgr" in band_full:
                            is_bgr = True
                            band = band_full.replace("_bgr", "")
                        else:
                            is_bgr = False
                            band = copy.deepcopy(band_full)

                        band_dir = os.path.join(
                            target_dir,
                            band_full,
                        )

                        if band_full not in progress_dict[target]:
                            progress_dict[target][band_full] = {
                                "success": True,
                                "data_moved": False,
                                "dir": None,
                                "run_astro_cat": False,
                            }

                        # Pull out the band type
                        band_type = get_band_type(band)

                        # Pull out whether we will do this step for this particular band
                        # and science type
                        do_step = True
                        if step_instrument is not None:
                            if step_instrument != band_type:
                                do_step = False

                        if step_science_type is not None:
                            if is_bgr and step_science_type == "sci":
                                do_step = False
                            if not is_bgr and step_science_type == "bgr":
                                do_step = False

                        if not do_step:
                            continue

                        # If we've failed elsewhere, skip here
                        if not progress_dict[target][band_full]["success"]:
                            continue

                        log.info(f"Beginning {step} for {band_full}")

                        # Pull out and make the directories we need
                        in_dir = copy.deepcopy(progress_dict[target][band_full]["dir"])
                        if in_dir is None:
                            # Pull the in band directory, else default to cal
                            if step in IN_BAND_DIRS:
                                in_band_dir = IN_BAND_DIRS[step]
                            else:
                                in_band_dir = "cal"

                            in_dir = os.path.join(
                                band_dir,
                                in_band_dir,
                            )

                        # If we need a specific out directory, pull it here.
                        # Else default to the name of the step
                        if step in OUT_BAND_DIRS:
                            out_band_dir = OUT_BAND_DIRS[step]
                        else:
                            out_band_dir = copy.deepcopy(step)

                        out_dir = os.path.join(
                            band_dir,
                            out_band_dir,
                        )

                        if not os.path.exists(in_dir):
                            os.makedirs(in_dir)
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)

                        # Move raw observations
                        if not progress_dict[target][band_full]["data_moved"]:
                            log.info(f"Moving raw observations for {band_full}")

                            if "move_raw_obs" in self.parameters:
                                move_raw_params = self.parameters["move_raw_obs"]
                            else:
                                move_raw_params = {}

                            kws = get_kws(
                                parameters=move_raw_params,
                                func=MoveRawObsStep,
                                target=target,
                                band=band,
                                max_level=1,
                            )

                            move_raw_obs = MoveRawObsStep(
                                target=target,
                                band=band,
                                step_ext=in_step_ext,
                                in_dir=self.raw_dir,
                                out_dir=in_dir,
                                is_bgr=is_bgr,
                                **kws,
                            )
                            step_result = move_raw_obs.do_step()

                            progress_dict[target][band_full]["success"] = copy.deepcopy(
                                step_result
                            )

                            # If we're not successful here, log a warning and delete the band folder,
                            # and move on
                            if not progress_dict[target][band_full]["success"]:
                                log.warning(
                                    f"Failures detected moving raw data for {target}, {band}. "
                                    f"Removing directories and continuing"
                                )
                                shutil.rmtree(out_dir)
                                continue

                            # Save out file moved state
                            progress_dict[target][band_full]["data_moved"] = True

                            log.info(f"Moved raw observations for {band_full}")

                        # Level 1 processing
                        if step == "lv1":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=Lv1Step,
                                target=target,
                                band=band,
                                max_level=0,
                            )

                            lv1 = Lv1Step(
                                target=target,
                                band=band,
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = lv1.do_step()

                        # Level 2 processing
                        elif step == "lv2":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=Lv2Step,
                                target=target,
                                band=band,
                                max_level=0,
                            )

                            lv2 = Lv2Step(
                                target=target,
                                band=band,
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                is_bgr=is_bgr,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = lv2.do_step()

                        elif step == "single_tile_destripe":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=SingleTileDestripeStep,
                                target=target,
                                band=band,
                            )

                            # If we're going from lv1, then this will be a rate file,
                            # otherwise a cal
                            if os.path.split(in_dir)[-1] == "lv1":
                                in_step_ext = "rate"

                            destripe = SingleTileDestripeStep(
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = destripe.do_step()

                        elif step == "lyot_mask":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=LyotMaskStep,
                                target=target,
                                band=band,
                            )

                            lyot_mask = LyotMaskStep(
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = lyot_mask.do_step()

                        elif step == "lyot_separate":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=LyotSeparateStep,
                                target=target,
                                band=band,
                            )

                            lyot_separate = LyotSeparateStep(
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = lyot_separate.do_step()

                        elif step == "multi_tile_destripe":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=MultiTileDestripeStep,
                                target=target,
                                band=band,
                            )

                            multi_tile_destripe = MultiTileDestripeStep(
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = multi_tile_destripe.do_step()

                        elif step == "apply_wcs_adjust":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=ApplyWCSAdjustStep,
                                target=target,
                                band=band,
                            )

                            wcs_adjust_file = os.path.join(
                                target_dir,
                                f"{target}_wcs_adjust.toml",
                            )
                            wcs_adjust = load_toml(wcs_adjust_file)

                            apply_wcs = ApplyWCSAdjustStep(
                                wcs_adjust=wcs_adjust,
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = apply_wcs.do_step()

                        elif step == "level_match":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=LevelMatchStep,
                                target=target,
                                band=band,
                            )

                            level_match = LevelMatchStep(
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = level_match.do_step()

                        elif step == "psf_model":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=PSFModelStep,
                                target=target,
                                band=band,
                            )

                            psf_model = PSFModelStep(
                                in_dir=in_dir,
                                out_dir=out_dir,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = psf_model.do_step()

                        elif step == "lv3":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=Lv3Step,
                                target=target,
                                band=band,
                                max_level=0,
                            )

                            lv3 = Lv3Step(
                                target=target,
                                band=band,
                                in_dir=in_dir,
                                out_dir=out_dir,
                                is_bgr=is_bgr,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = lv3.do_step()

                        elif step == "astrometric_catalog":
                            kws = get_kws(
                                parameters=step_parameters,
                                func=AstrometricCatalogStep,
                                target=target,
                                band=band,
                                max_level=0,
                            )

                            astrometric_catalog = AstrometricCatalogStep(
                                target=target, band=band, in_dir=in_dir, **kws
                            )
                            step_result = astrometric_catalog.do_step()

                            progress_dict[target][band_full]["run_astro_cat"] = True

                        elif step == "astrometric_align":
                            # If we've run the astrometric catalog step, track
                            # that here
                            run_astro_cat = copy.deepcopy(
                                progress_dict[target][band_full]["run_astro_cat"]
                            )

                            kws = get_kws(
                                parameters=step_parameters,
                                func=AstrometricAlignStep,
                                target=target,
                                band=band,
                                max_level=0,
                            )

                            astrometric_catalog = AstrometricAlignStep(
                                target=target,
                                band=band,
                                target_dir=target_dir,
                                in_dir=in_dir,
                                is_bgr=is_bgr,
                                catalog_dir=self.alignment_dir,
                                run_astro_cat=run_astro_cat,
                                step_ext=in_step_ext,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = astrometric_catalog.do_step()

                        elif step == "mosaic_individual_fields":
                            # Here, the input directory should be level 3
                            mosaic_in_dir = os.path.join(
                                band_dir,
                                "lv3",
                            )

                            kws = get_kws(
                                parameters=step_parameters,
                                func=MosaicIndividualFieldsStep,
                                target=target,
                                band=band,
                                max_level=0,
                            )

                            mosaic_individual_fields = MosaicIndividualFieldsStep(
                                target=target,
                                band=band,
                                in_dir=mosaic_in_dir,
                                out_dir=out_dir,
                                procs=self.procs,
                                **kws,
                            )
                            step_result = mosaic_individual_fields.do_step()

                        elif step == "psf_matching":

                            psf_matching = PSFMatchingStep(
                                target=target,
                                band=band,
                                in_dir=in_dir,
                                out_dir=out_dir,
                                kernel_dir=self.kernel_dir,
                                in_step_ext=in_step_ext,
                                procs=self.procs,
                                **step_parameters,
                            )
                            step_result = psf_matching.do_step()

                        else:
                            raise ValueError(
                                f"step should be one of {ALLOWED_STEPS}, not {step}"
                            )

                        progress_dict[target][band_full]["success"] = copy.deepcopy(
                            step_result
                        )
                        progress_dict[target][band_full]["dir"] = copy.deepcopy(out_dir)

                        # If we're not successful here, log a warning and delete the band folder
                        if not progress_dict[target][band_full]["success"]:
                            log.warning(
                                f"Failures detected in step {step} for {target}, {band_full}. "
                                f"Removing folder and continuing"
                            )
                            shutil.rmtree(band_dir)

                        log.info(f"Completed {step} for {band_full}")
