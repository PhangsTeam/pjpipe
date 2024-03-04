import copy
import glob
import logging
import os

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from photutils.detection import DAOStarFinder, IRAFStarFinder

from ..utils import parse_parameter_dict, fwhms_pix, sigma_clip, recursive_setattr

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())

ALLOWED_STARFIND_METHODS = [
    "dao",
    "iraf",
]


class AstrometricCatalogStep:
    def __init__(
            self,
            target,
            band,
            in_dir,
            snr=5,
            starfind_method='dao',
            starfind_parameters=None,
            dao_parameters=None,
            overwrite=False,
    ):
        """Generate a catalog for absolute astrometric alignment

        Args:
            in_dir: Directory to search for files
            snr: SNR to detect sources. Defaults to 5
            starfind_method: Method for detecting sources in image. Options are given be
                ALLOWED_STARFIND_METHODS
            starfind_parameters: Dictionary of parameters to pass to the starfinder
            dao_parameters: Dictionary of parameters to pass to DAOFinder
            overwrite: Overwrite or not. Defaults to False
        """

        if starfind_method not in ALLOWED_STARFIND_METHODS:
            raise ValueError(f"starfind_method should be one of {ALLOWED_STARFIND_METHODS}")

        if dao_parameters is not None:
            log.warning("dao_parameters has been deprecated in favour of starfind_parameters, "
                        "and will fail in the future")
            starfind_parameters = copy.deepcopy(dao_parameters)

        if starfind_parameters is None:
            starfind_parameters = {}

        self.in_dir = in_dir
        self.target = target
        self.band = band
        self.snr = snr
        self.starfind_method = starfind_method
        self.starfind_parameters = starfind_parameters
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

        if self.overwrite:
            os.system(f"rm -rf {step_complete_file}")

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
        """Generate an astrometric catalogue using given starfinder

        Args:
            file: File to run starfinder on
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

        if self.starfind_method == "dao":
            finder = DAOStarFinder
        elif self.starfind_method == "iraf":
            finder = IRAFStarFinder
        else:
            raise ValueError(f"starfind_method should be one of {ALLOWED_STARFIND_METHODS}")

        starfind = finder(
            fwhm=kernel_fwhm,
            threshold=threshold,
        )

        for astro_key in self.starfind_parameters:
            value = parse_parameter_dict(
                self.starfind_parameters,
                astro_key,
                self.band,
                self.target,
            )

            if value == "VAL_NOT_FOUND":
                continue

            recursive_setattr(starfind, astro_key, value)

        sources = starfind(data, mask=mask)

        # Add in RA and Dec
        ra, dec = w.all_pix2world(sources["xcentroid"], sources["ycentroid"], 0)
        sky_coords = SkyCoord(ra * u.deg, dec * u.deg)
        sources.add_column(sky_coords, name="sky_centroid")
        sources.write(cat_name, overwrite=True)

        return True
