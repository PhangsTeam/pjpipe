import glob
import logging
import os

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder

from ..utils import parse_parameter_dict, fwhms_pix, sigma_clip, recursive_setattr

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


class AstrometricCatalogStep:
    def __init__(
        self,
        target,
        band,
        in_dir,
        snr=5,
        dao_parameters=None,
        overwrite=False,
    ):
        """Generate a catalog for absolute astrometric alignment

        Args:
            in_dir: Directory to search for files
            snr: SNR to detect sources. Defaults to 5
            dao_parameters: Dictionary of parameters to pass to DAOFinder
            overwrite: Overwrite or not. Defaults to False
        """

        if dao_parameters is None:
            dao_parameters = {}

        self.in_dir = in_dir
        self.target = target
        self.band = band
        self.snr = snr
        self.dao_parameters = dao_parameters
        self.overwrite = overwrite

    def do_step(self):
        """Run astrometric catalog step"""

        if self.overwrite:
            os.system(f"rm -rf {os.path.join(self.in_dir, '*_astro_cat.fits')}")

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.in_dir,
            "astrometric_catalog_step_complete.txt",
        )
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        jwst_files = glob.glob(
            os.path.join(
                self.in_dir,
                "*i2d.fits",
            )
        )

        successes = []
        for jwst_file in jwst_files:
            success = self.generate_astro_cat(jwst_file)
            successes.append(success)

        if not np.all(successes):
            log.warning("Failures detected in astrometric catalog step")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def generate_astro_cat(
        self,
        file,
    ):
        """Generate an astrometric catalogue using DAOStarFinder

        Args:
            file: File to run DAOStarFinder on
        """

        log.info(f"Creating astrometric catalog for {file}")

        cat_name = file.replace("_i2d.fits", "_astro_cat.fits")

        with fits.open(file, memmap=False) as hdu:
            data_hdu = hdu["SCI"]
            w = WCS(data_hdu)
            data = data_hdu.data

        del hdu

        snr = self.snr

        mask = data == 0
        mean, median, rms = sigma_clip(data, dq_mask=mask)
        threshold = median + snr * rms

        kernel_fwhm = fwhms_pix[self.band]

        daofind = DAOStarFinder(
            fwhm=kernel_fwhm,
            threshold=threshold,
        )

        for astro_key in self.dao_parameters:
            value = parse_parameter_dict(
                self.dao_parameters,
                astro_key,
                self.band,
                self.target,
            )

            if value == "VAL_NOT_FOUND":
                continue

            recursive_setattr(daofind, astro_key, value)

        sources = daofind(data, mask=mask)

        # Add in RA and Dec
        ra, dec = w.all_pix2world(sources["xcentroid"], sources["ycentroid"], 0)
        sky_coords = SkyCoord(ra * u.deg, dec * u.deg)
        sources.add_column(sky_coords, name="sky_centroid")
        sources.write(cat_name, overwrite=True)

        return True
