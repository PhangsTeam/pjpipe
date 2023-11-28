import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
import warnings
from functools import partial

import numpy as np
from jwst.assign_wcs.util import update_fits_wcsinfo
from stdatamodels.jwst import datamodels
from tqdm import tqdm
from tweakwcs.correctors import JWSTWCSCorrector

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class ApplyWCSAdjustStep:
    def __init__(
        self,
        wcs_adjust,
        in_dir,
        out_dir,
        step_ext,
        procs,
        overwrite=False,
    ):
        """Apply WCS adjustments to images

        Args:
            wcs_adjust: Dictionary for WCS adjustments
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        self.wcs_adjust = wcs_adjust
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs
        self.overwrite = overwrite

    def do_step(self):
        """Run applying the WCS adjustments"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "apply_wcs_adjust_step_complete.txt",
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

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(files)])

        successes = self.run_step(
            files,
            procs=procs,
        )

        if not np.all(successes):
            log.warning("Failures detected in applying WCS adjustments")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(
        self,
        files,
        procs=1,
    ):
        """Wrap paralellism around applying WCS adjusts

        Args:
            files: List of files to mask lyot in
            procs: Number of parallel processes to run.
                Defaults to 1
        """

        log.info(f"Applying WCS corrections")

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in tqdm(
                pool.imap_unordered(
                    partial(
                        self.parallel_wcs_adjust,
                    ),
                    files,
                ),
                ascii=True,
                desc="Applying WCS corrections",
                total=len(files),
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_wcs_adjust(
        self,
        file,
    ):
        """Parallelise applying WCS adjustments

        Args:
            file: File to apply WCS corrections to
        """

        file_short = os.path.split(file)[-1]

        output_file = os.path.join(
            self.out_dir,
            file_short,
        )

        # Set up the WCSCorrector per tweakreg
        with datamodels.open(file) as input_im:
            ref_wcs = input_im.meta.wcs
            ref_wcsinfo = input_im.meta.wcsinfo.instance

            im = JWSTWCSCorrector(ref_wcs, ref_wcsinfo)

            # Pull out the info we need to shift. If we have both
            # dithers ungrouped and grouped, prefer the ungrouped
            # ones
            visit_grouped = file_short.split("_")[0]
            visit_ungrouped = "_".join(file_short.split("_")[:3])

            matrix = [[1, 0], [0, 1]]
            shift = [0, 0]

            visit_found = False
            for visit in [visit_ungrouped, visit_grouped]:
                if not visit_found:
                    if visit in self.wcs_adjust["wcs_adjust"]:
                        wcs_adjust_vals = self.wcs_adjust["wcs_adjust"][visit]

                        try:
                            matrix = wcs_adjust_vals["matrix"]
                        except KeyError:
                            matrix = [[1, 0], [0, 1]]

                        try:
                            shift = wcs_adjust_vals["shift"]
                        except KeyError:
                            shift = [0, 0]

                        visit_found = True

            if not visit_found:
                log.info(f"No shifts found for {file_short}. Defaulting to no shift")

            im.set_correction(matrix=matrix, shift=shift)

            input_im.meta.wcs = im.wcs

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    update_fits_wcsinfo(
                        input_im,
                    )
                except (ValueError, RuntimeError) as e:
                    log.warning(
                        "Failed to update 'meta.wcsinfo' with FITS SIP "
                        f"approximation. Reported error is:\n'{e.args[0]}'"
                    )

            input_im.save(output_file)

        del input_im
        del im
        gc.collect()

        return True
