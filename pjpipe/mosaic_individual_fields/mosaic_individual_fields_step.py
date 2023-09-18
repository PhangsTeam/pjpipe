import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
from functools import partial

import numpy as np
from jwst.resample import ResampleStep
from stdatamodels.jwst import datamodels

from ..utils import parse_parameter_dict, recursive_setattr

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class MosaicIndividualFieldsStep:
    def __init__(
        self,
        target,
        band,
        in_dir,
        out_dir,
        procs,
        crf_ext="crf_tweakback",
        resample_parameters=None,
        parallel=True,
        overwrite=False,
    ):
        """Mosaic each individual field in an observation set

        N.B. This should be run on _crf_tweakback files, so
        this step assumes you've already run level 3 and any
        astrometric alignment.

        Args:
            target: Target to consider
            band: Band to consider
            in_dir: Input directory for the crf files
            out_dir: Where to save the mosaicked files to
            procs: Number of processes to run in parallel
            crf_ext: Extension for files to mosaic. Defaults
                to crf_tweakback
            resample_parameters: Parameters to pass to the
                JWST resample step
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        if resample_parameters is None:
            resample_parameters = {}

        if procs == 1:
            parallel = False

        self.target = target
        self.band = band
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.procs = procs
        self.crf_ext = crf_ext
        self.resample_parameters = resample_parameters
        self.parallel = parallel
        self.overwrite = overwrite

    def do_step(self):
        """Run mosaicking individual fields"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "mosaic_individual_fields_step_complete.txt",
        )
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        # Figure out the individual visits we have
        all_files = glob.glob(
            os.path.join(
                self.in_dir,
                f"*_{self.crf_ext}.fits",
            )
        )
        short_files = [os.path.split(file)[-1] for file in all_files]
        individual_visits = np.unique(
            ["_".join(file.split("_")[:2]) for file in short_files]
        )

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(individual_visits)])

        successes = self.run_step(
            individual_visits,
            procs=procs,
        )

        # If not everything has succeeded, then return a warning
        if not np.all(successes):
            log.warning("Failures detected in individual field mosaicking")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(
        self,
        individual_visits,
        procs=1,
    ):
        """Wrap parallelism around mosaicking

        Args:
            individual_visits: List of individual visits
            procs: Number of parallel processes to run. Defaults
                to 1
        """

        successes = []

        if self.parallel:
            with mp.get_context("fork").Pool(procs) as pool:
                for success in pool.imap_unordered(
                    partial(
                        self.parallel_mosaic_fields,
                    ),
                    individual_visits,
                ):
                    successes.append(success)

                pool.close()
                pool.join()

        else:
            for individual_visit in individual_visits:
                successes.append(self.parallel_mosaic_fields(individual_visit))

        gc.collect()

        return successes

    def parallel_mosaic_fields(
        self,
        visit,
    ):
        """Parallellise mosaicking

        Args:
            visit: Visit to produce mosaic for
        """

        input_files = glob.glob(
            os.path.join(self.in_dir, f"{visit}_*_{self.crf_ext}.fits"),
        )
        output_file = os.path.join(self.out_dir, f"{visit}_indiv_field.fits")
        input_files.sort()
        input_models = [datamodels.open(input_file) for input_file in input_files]
        config = ResampleStep.get_config_from_reference(input_models)
        step = ResampleStep.from_config_section(config)

        # Set any optional parameters
        for key in self.resample_parameters:
            value = parse_parameter_dict(
                parameters=self.resample_parameters,
                key=key,
                band=self.band,
                target=self.target,
            )

            if value == "VAL_NOT_FOUND":
                continue

            recursive_setattr(step, key, value)

        step.save_results = False
        output_model = step.run(input_models)
        output_model.save(output_file)
        del input_models, output_model

        return True
