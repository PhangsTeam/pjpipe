import gc
import glob
import logging
import multiprocessing as mp
import os
import shutil
from functools import partial
from astropy.io import fits
from astropy.wcs import WCS


import numpy as np

from ..utils import do_jwst_convolution

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class PSFMatchingStep:
    def __init__(
        self,
        target,
        in_dir,
        out_dir,
        kernel_dir,
        procs,
        in_step_ext,
        band=None,
        target_bands=None,
        overwrite=False,
    ):
        """Match PSF for all images

        Taking a list of target resolutions and kernels, will convolve to those resolutions.
        If an existing file already exists (e.g. a F2100W image from processing), will also
        regrid to that pixel grid

        Args:
            target: Target to consider
            in_dir: Input directory
            out_dir: Output directory
            kernel_dir: Kernel directory
            procs: Number of processes to run in parallel
            in_step_ext: Filename extension for the input files
            band: Bands to consider
            target_bands: Bands to convolve to
            overwrite: Whether to overwrite or not
        """

        if kernel_dir is None or not os.path.exists(kernel_dir):
            raise ValueError("kernel_dir should be defined and should exist")

        self.target = target
        self.band = band
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.kernel_dir = kernel_dir
        self.procs = procs
        self.in_step_ext = in_step_ext
        self.target_bands = target_bands
        self.overwrite = overwrite

    def do_step(self):
        """Run psf_matching step"""

        step_complete_file = os.path.join(
            self.out_dir,
            "psf_matching_step_complete.txt",
        )

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        # Check if we've already run the step
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        files = glob.glob(
            os.path.join(
                self.in_dir,
                f"*_{self.in_step_ext}.fits",
            )
        )
        files.sort()

        # If we don't have anything, warn but succeed
        if len(files) == 0:
            log.warning("No files found, will skip this step")
            with open(step_complete_file, "w+") as f:
                f.close()
            return True

        procs = np.nanmin([self.procs, len(files) * len(self.target_bands)])

        successes = self.run_step(files, procs=procs)

        if not np.all(successes):
            log.warning("Failures detected during PSF matching")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def run_step(self, files, procs=1):
        """Wrap paralellism around applying psf matching

        Args:
            files: List of files to process
            procs: Number of parallel processes to run.
                Defaults to 1
        """

        log.info("Running PSF matching")

        files_process = []
        target_band_process = []
        for f in files:
            files_process.extend([f] * len(self.target_bands))
            target_band_process.extend(self.target_bands)

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in pool.imap_unordered(
                partial(
                    self.parallel_psf_match,
                    current_band=self.band,
                ),
                zip(files_process, target_band_process),
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_psf_match(self, current_task, current_band=None):
        """Parallelize psf matching to target resolution

        Args:
            current_task: tuple (file, target_band),
                where file is the File to apply psf matching,
                and target_band is the band of target resolution
            current_band: band of the current image

        Returns:
            True or False
        """
        file, target_band = current_task
        file_short = os.path.split(file)[-1]
        file_short = file_short.replace(
            self.in_step_ext, f"{self.in_step_ext}_at{target_band}"
        )
        output_file = os.path.join(self.out_dir, file_short)
        kernel_file = os.path.join(
            self.kernel_dir, f"{current_band.lower()}_to_{target_band.lower()}.fits"
        )
        if not os.path.exists(kernel_file):
            raise FileNotFoundError(
                f"Kernel file {os.path.split(kernel_file)[-1]} not found"
            )

        # If this is a JWST band, then we want to reproject to match the pixel grid of that existing
        # image
        check_file = file.replace(current_band.upper(), target_band.upper())
        check_file = check_file.replace(current_band.lower(), target_band.lower())
        if os.path.exists(check_file):
            with fits.open(check_file) as hdu:
                output_grid = (WCS(hdu["SCI"].header), hdu["SCI"].data.shape)
        else:
            output_grid = None

        do_jwst_convolution(
            file, output_file, file_kernel=kernel_file, output_grid=output_grid
        )

        return True
