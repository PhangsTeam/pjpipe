import copy
import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
from functools import partial

import numpy as np
from reproject import reproject_interp
from stdatamodels.jwst import datamodels
from stdatamodels.jwst.datamodels.dqflags import pixel
from tqdm import tqdm

from ..utils import get_dq_bit_mask

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

ALLOWED_METHODS = ["mask", "mask_overlap"]

# Rough lyot outline
LYOT_I = slice(735, None)
LYOT_J = slice(None, 290)


class LyotMaskStep:
    def __init__(
        self,
        in_dir,
        out_dir,
        step_ext,
        procs,
        method="mask",
        overwrite=False,
    ):
        """Mask the lyot coronagraph in MIRI observations

        Args:
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel
            method: Whether to just mask the coronagraph (mask),
                or only parts that overlap the main science
                chip in other observations (mask_overlap).
                Defaults to 'mask'
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        if method not in ALLOWED_METHODS:
            raise ValueError(f"method should be one of {ALLOWED_METHODS}")

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs
        self.method = method
        self.overwrite = overwrite

    def do_step(self):
        """Run lyot masking"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)
            os.system(f"rm -rf {os.path.join(self.in_dir, '*.json')}")

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "lyot_mask_step_complete.txt",
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
        """Wrap paralellism around the lyot masking

        Args:
            files: List of files to mask lyot in
            procs: Number of parallel processes to run.
                Defaults to 1
        """

        # Pull out the value we'll mask the DQ array to
        mask_value = pixel["DO_NOT_USE"] + pixel["NON_SCIENCE"]

        log.info(f"Running lyot masking with method {self.method}")

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            if self.method == "mask":
                for success in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_lyot_mask,
                            mask_value=mask_value,
                        ),
                        files,
                    ),
                    ascii=True,
                    desc="Masking lyot region",
                    total=len(files),
                ):
                    successes.append(success)
            elif self.method == "mask_overlap":
                for success in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_lyot_mask_overlap,
                            all_files=files,
                            mask_value=mask_value,
                        ),
                        files,
                    ),
                    ascii=True,
                    desc="Masking lyot overlap region",
                    total=len(files),
                ):
                    successes.append(success)
            else:
                raise ValueError(f"method should be one of {ALLOWED_METHODS}")

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_lyot_mask(
        self,
        file,
        mask_value=513,
    ):
        """Parallelise lyot masking

        Args:
            file: File to mask lyot in
            mask_value: DQ bit value for masked values.
                Defaults to 513 (DO_NOT_USE+NON_SCIENCE)
        """

        short_file = os.path.split(file)[-1]
        out_file = os.path.join(
            self.out_dir,
            short_file,
        )

        with datamodels.open(file) as im:
            # If we have subarray data, don't mask
            is_subarray = "sub" in im.meta.subarray.name.lower()
            if is_subarray:
                im.save(out_file)
                del im
                return True

            # Get a rough mask to start
            lyot_rough_mask = np.zeros_like(im.dq)
            lyot_rough_mask[LYOT_I, LYOT_J] = 1

            # Get DQ mask
            dq_mask = get_dq_bit_mask(im.dq)

            # The final mask is "science" data within
            # the lyot region
            lyot_mask = np.logical_and(
                dq_mask == 0,
                lyot_rough_mask == 1,
            )
            im.dq[lyot_mask] = mask_value

            im.save(out_file)

            del im

        return True

    def parallel_lyot_mask_overlap(
        self,
        file,
        all_files,
        mask_value=513,
    ):
        """Parallelise lyot overlap masking

        Args:
            file: File to mask lyot in
            all_files: Complete list of files
            mask_value: DQ bit value for masked values.
                Defaults to 513 (DO_NOT_USE+NON_SCIENCE)
        """

        short_file = os.path.split(file)[-1]
        out_file = os.path.join(
            self.out_dir,
            short_file,
        )

        all_file_info = []
        for all_file in all_files:
            with datamodels.open(all_file) as im:
                # Pull out the WCS, and get a science mask (non-lyot)
                # for each image
                wcs = copy.deepcopy(im.meta.wcs.to_fits_sip())
                dq = get_dq_bit_mask(im.dq).astype(bool)
                dq[LYOT_I, LYOT_J] = 1

                all_file_info.append([wcs, ~dq, all_file])
                del im

        with datamodels.open(file) as im:
            im_shape = im.data.shape
            im_wcs = im.meta.wcs.to_fits_sip()

            lyot_rough_mask = np.zeros_like(im.data)
            lyot_rough_mask[LYOT_I, LYOT_J] = 1

            # Get DQ mask
            dq_mask = get_dq_bit_mask(im.dq)

            # The final mask is "science" data within
            # the lyot region
            lyot_mask = np.logical_and(
                dq_mask == 0,
                lyot_rough_mask == 1,
            )

            # We now loop over each file, reprojecting the science
            # (non-lyot) data to see if the lyot is overlapping other
            # science regions
            cumulative_mask = np.zeros_like(im.data, dtype=bool)

            for file_info in all_file_info:
                if file == file_info[-1]:
                    continue

                file_wcs = copy.deepcopy(file_info[0])
                file_science_mask = copy.deepcopy(file_info[1])

                # Reproject the science mask
                rd = reproject_interp(
                    input_data=(file_science_mask, file_wcs),
                    output_projection=im_wcs,
                    shape_out=im_shape,
                    order="nearest-neighbor",
                    return_footprint=False,
                )

                # For pixels within the lyot mask and also science data of another
                # image, mask now!
                cumulative_mask[np.logical_and(lyot_mask == 1, rd == 1)] = True
            im.dq[cumulative_mask] = mask_value

            im.save(out_file)

            del im

        return True
