import copy
import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
from functools import partial

import numpy as np
from jwst.pipeline import calwebb_detector1

from ..utils import attribute_setter

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class Lv1Step:
    def __init__(
        self,
        target,
        band,
        in_dir,
        out_dir,
        step_ext,
        procs,
        jwst_parameters=None,
        overwrite=False,
    ):
        """Wrapper around the level 1 JWST pipeline

        Args:
            target: Target to consider
            band: Band to consider
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel.
            jwst_parameters: Parameter dictionary to pass to
                the level 1 pipeline. Defaults to None,
                which will run the observatory defaults
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        if jwst_parameters is None:
            jwst_parameters = {}

        self.target = target
        self.band = band
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs
        self.jwst_parameters = jwst_parameters
        self.overwrite = overwrite

    def do_step(self):
        """Run the level 1 pipeline"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "lv1_step_complete.txt",
        )
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        # We need to operate this in the input directory
        cwd = os.getcwd()
        os.chdir(self.in_dir)

        # Build file list
        in_files = glob.glob(f"*_{self.step_ext}.fits")

        if len(in_files) == 0:
            log.warning(f"No {self.step_ext} files found")
            os.chdir(cwd)
            return False

        in_files.sort()

        # For speed, we want to parallelise these up by dither since we use the
        # persistence file
        dithers = []
        for file in in_files:
            file_split = os.path.split(file)[-1].split("_")
            dithers.append("_".join(file_split[:2]) + "_*_" + file_split[-2])
        dithers = np.unique(dithers)
        dithers.sort()

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(dithers)])

        successes = self.run_step(
            dithers,
            procs=procs,
        )

        # If not everything has succeeded, then return a warning
        if not np.all(successes):
            log.warning("Failures detected in level 1 pipeline")
            os.chdir(cwd)
            return False

        os.chdir(cwd)

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(
        self,
        dithers,
        procs=1,
    ):
        """Wrap parallelism around the level 1 pipeline

        Args:
            dithers: List of dithers to loop over
            procs: Number of processes to run. Defaults to 1
        """

        log.info("Running level 1 pipeline")

        # Ensure we pre-cache references, to avoid errors in multiprocessing. Loop over
        # all just to be safe
        for dither in dithers:
            uncal_files = glob.glob(f"{dither}*_{self.step_ext}.fits")
            for uncal_file in uncal_files:
                config = calwebb_detector1.Detector1Pipeline.get_config_from_reference(
                    uncal_file
                )
                detector1 = calwebb_detector1.Detector1Pipeline.from_config_section(
                    config
                )
                detector1._precache_references(uncal_file)

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in pool.imap_unordered(
                partial(
                    self.parallel_lv1,
                ),
                dithers,
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_lv1(
        self,
        dither,
    ):
        """Parallelise lv1 reprocessing

        Args:
            dither: Name for dither group. This is used because
                we inherit persistence from previous integration
                in the set
        """

        uncal_files = glob.glob(f"{dither}*_{self.step_ext}.fits")
        uncal_files.sort()

        for uncal_file in uncal_files:
            config = calwebb_detector1.Detector1Pipeline.get_config_from_reference(
                uncal_file
            )
            detector1 = calwebb_detector1.Detector1Pipeline.from_config_section(config)

            # Pull out the trapsfilled file from preceding exposure
            persist_file = ""

            uncal_file_split = uncal_file.split("_")
            exposure_str = uncal_file_split[2]
            prev_exposure_int = int(exposure_str) - 1

            if prev_exposure_int > 0:
                prev_exposure_str = f"{prev_exposure_int:05}"
                persist_file = copy.deepcopy(uncal_file_split)
                persist_file[2] = prev_exposure_str
                persist_file[-1] = "trapsfilled.fits"
                persist_file = os.path.join(self.out_dir, "_".join(persist_file))

            # Specify the name of the trapsfilled file
            detector1.persistence.input_trapsfilled = persist_file

            # Set other parameters
            detector1.output_dir = self.out_dir

            detector1 = attribute_setter(
                detector1,
                parameters=self.jwst_parameters,
                band=self.band,
                target=self.target,
            )

            # Run the level 1 pipeline
            detector1.run(uncal_file)

            del detector1
            gc.collect()

        return True
