import copy
import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
from functools import partial

import numpy as np
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels.dqflags import pixel
from tqdm import tqdm

from ..utils import get_dq_bit_mask

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

# Rough lyot outline
LYOT_I = slice(735, None)
LYOT_J = slice(None, 290)


class LyotSeparateStep:
    def __init__(
        self,
        in_dir,
        out_dir,
        step_ext,
        procs,
        miri_ext="mirimage",
        overwrite=False,
    ):
        """Separate each MIRI file out into main science chip and lyot coronagraph

        Args:
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel
            miri_ext: MIRI filename extension. Defaults to "mirimage"
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs
        self.miri_ext = miri_ext
        self.overwrite = overwrite

    def do_step(self):
        """Run lyot separation"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)
            os.system(f"rm -rf {os.path.join(self.in_dir, '*.json')}")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "lyot_separate_step_complete.txt",
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

        # If not everything has succeeded, then return a warning
        if not np.all(successes):
            log.warning("Failures detected in level 2 pipeline")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(
        self,
        files,
        procs=1,
    ):
        """Wrap paralellism around the lyot separation

        Args:
            files: List of files to separate lyot in
            procs: Number of parallel processes to run.
                Defaults to 1
        """

        log.info(f"Running lyot separation")

        # Pull out the value we'll mask the DQ array to
        mask_value = pixel["DO_NOT_USE"] + pixel["NON_SCIENCE"]

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in tqdm(
                pool.imap_unordered(
                    partial(
                        self.parallel_lyot_separate,
                        mask_value=mask_value,
                    ),
                    files,
                ),
                ascii=True,
                desc="Separating lyot region",
                total=len(files),
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_lyot_separate(
        self,
        file,
        mask_value=513,
    ):
        """Parallelise lyot separation

        Args:
            file: File to separate lyot in
            mask_value: DQ bit value for masked values.
                Defaults to 513 (DO_NOT_USE+NON_SCIENCE)
        """

        short_file = os.path.split(file)[-1]

        # Pull out filenames for the main chip image
        # and lyot. These need to be distinct, but the
        # same length. We do this by distinguishing the
        # main science as miri_ext+"s", and the lyot as
        # miri_ext+"l"
        main_chip_out_file = os.path.join(
            self.out_dir,
            short_file.replace(self.miri_ext, f"{self.miri_ext}s"),
        )

        lyot_out_file = os.path.join(
            self.out_dir,
            short_file.replace(self.miri_ext, f"{self.miri_ext}l"),
        )

        with datamodels.open(file) as im:
            lyot_rough_mask = np.zeros_like(im.data)
            lyot_rough_mask[LYOT_I, LYOT_J] = 1

            # If we have subarray data, don't separate things out
            is_subarray = "sub" in im.meta.subarray.name.lower()
            if is_subarray:
                im.save(main_chip_out_file)
                del im
                return True

            # Get DQ mask
            dq_mask = get_dq_bit_mask(im.dq)

            # Create 2 masks for the lyot and the main
            # science chip
            lyot_mask = np.logical_and(
                dq_mask == 0,
                lyot_rough_mask == 1,
            )
            main_chip_mask = np.logical_and(
                dq_mask == 0,
                lyot_rough_mask == 0,
            )

            main_chip_im = copy.deepcopy(im)
            lyot_im = copy.deepcopy(im)

            # Mask lyot in main chip, and main
            # chip in lyot
            main_chip_im.dq[lyot_mask] = mask_value
            lyot_im.dq[main_chip_mask] = mask_value

            main_chip_im.save(main_chip_out_file)
            lyot_im.save(lyot_out_file)

            del main_chip_im, lyot_im, im

        return True
