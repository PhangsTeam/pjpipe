import copy
import gc
import glob
import itertools
import logging
import multiprocessing as mp
import os
import random
import shutil
import warnings
from functools import partial

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from reproject.mosaicking import find_optimal_celestial_wcs
from scipy.optimize import minimize
from stdatamodels.jwst import datamodels
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from ..utils import get_dq_bit_mask, reproject_image, make_source_mask, make_stacked_image, get_band_type

# Rough lyot outline
LYOT_I = slice(735, None)
LYOT_J = slice(None, 290)

ALLOWED_REPROJECT_FUNCS = [
    "interp",
    "adaptive",
    "exact",
]

ALLOWED_FIT_TYPES = [
    "level",
    "level+slope",
]

matplotlib.use("agg")
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

log = logging.getLogger("stpipe")
log.addHandler(logging.NullHandler())


def get_dithers(files,
                combine_nircam_short=False,
                ):
    """Get unique dithers from a list of files

    Args:
        files: List of input files
        combine_nircam_short: Whether to drop the chip number from
            NIRCam short observations. Defaults to False
    """

    # Split these into dithers per-chip
    dithers = []
    for file in files:
        file_split = os.path.split(file)[-1].split("_")
        dither_str = "_".join(file_split[:2]) + "_*_" + file_split[-2]
        if combine_nircam_short:
            dither_str = dither_str[:-1] + "*"
        dithers.append(dither_str)
    dithers = np.unique(dithers)
    dithers.sort()

    return dithers


def apply_subtraction(im,
                      delta,
                      fit_type="level",
                      ref_ra=0,
                      ref_dec=0,
                      ref_wcs=None,
                      ref_shape=None,
                      ):
    """Apply subtraction to the image.

    Args:
        im: Input datamodel
        delta: Coefficients to subtract from the image
        fit_type: Type of fitting we've done. Options
            are 'level' (the default, just fits a
            single offset), and 'level+slope' (which will
            fit a plane)
        ref_ra: The reference RA point for the fits. Defaults to 0
        ref_dec: The reference Dec point for the fits. Defaults to 0
        ref_wcs: Reference WCS for the fits, since we do in pixel space.
            If None, will assume this is the WCS for the image, which is
            likely incorrect
        ref_shape: Shape for the reference WCS, since we do the fits in
            pixel space. If None, will assume the image shape, which is
            likely incorrect
    """

    zero_idx = im.data == 0

    if fit_type == "level":
        im.data -= delta[0]
    elif fit_type == "level+slope":

        # Here, pull out the WCS and use this to convert the delta
        # coefficients to a slope for valid pixels
        w = copy.deepcopy(im.meta.wcs.to_fits_sip())
        wcs = WCS(w)

        ii, jj = np.indices(im.data.shape, dtype=float)

        if ref_wcs is None:
            ref_wcs = copy.deepcopy(wcs)
        if ref_shape is None:
            ref_shape = copy.deepcopy(im.data.shape)

        # This is a little fiddly, we need to convert to the proper reference frame and then map pixel
        # coordinates
        frame_coords = wcs.pixel_to_world(jj, ii)

        ii_ref, jj_ref = np.indices(ref_shape)
        ref_coords = ref_wcs.pixel_to_world(jj_ref, ii_ref)

        # Convert the input world coordinates to the frame of the output world
        # coordinates.
        frame_coords = frame_coords.transform_to(ref_coords.frame)

        # Compute the pixel positions in the *output* image of the pixels
        # from the *input* image.
        jj, ii = ref_wcs.world_to_pixel(frame_coords)

        # Finally, subtract off the reference
        ref_j, ref_i = get_x_y_values(ref_wcs, ref_ra, ref_dec)
        jj -= ref_j
        ii -= ref_i

        delta_plane = delta[0] * jj + delta[1] * ii + delta[2]

        im.data -= delta_plane

    else:
        raise ValueError(f"fit_type {fit_type} not known, should be one of {ALLOWED_FIT_TYPES}")

    im.data[zero_idx] = 0

    return im


def plane(x, y, params):
    """Define a plane of the form

    params[0] * x + params[1] * y + params[2]

    Args:
        x: x coordinates
        y: y coordinates
        params: Parameters for the plane
    """

    a = params[0]
    b = params[1]
    c = params[2]
    z = a * x + b * y + c
    return z


def plane_resid(params,
                points,
                err=None,
                return_sum=True,
                ):
    """Calculates the difference between a plane and input points

    Args:
        params: Parameters for the plane
        points: (x, y, z) coordinates measured
        err: Error on each z-point. Defaults to None
        return_sum: Whether to return the sum or all
            the individual values. Defaults to True
    """

    plane_z = plane(points[:, 0], points[:, 1], params)
    diff = points[:, 2] - plane_z

    if err is None:
        result = diff ** 2
    else:
        result = diff ** 2 / err ** 2

    # Scale chisq using sqrt(2N) to account for large number of points
    result /= np.sqrt(2 * len(points[:, 0]))

    if return_sum:
        result = np.nansum(result)

    return result


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def print_delta(file,
                delta,
                fit_type="level",
                ):
    """Format the delta for the log properly

    Args:
        file: Filename to apply the delta to
        delta: Delta coefficients
        fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
    """

    if fit_type == "level":
        log.info(f"{file}, delta={delta[0]:.2e}")
    elif fit_type == "level+slope":

        # Format pos/negs correctly in the string
        if delta[1] < 0:
            sign_dec = "-"
        else:
            sign_dec = "+"
        if delta[2] < 0:
            sign_dc = "-"
        else:
            sign_dc = "+"

        log.info(f"{file}, plane={delta[0]:.2e} * x (pix) "
                 f"{sign_dec} {np.abs(delta[1]):.2e} * y (pix) "
                 f"{sign_dc} {np.abs(delta[2]):.2e}")
    else:
        raise ValueError(f"fit_type should be one of {ALLOWED_FIT_TYPES}")


def get_ra_dec_values(wcs,
                      jj,
                      ii,
                      units=u.arcsec,
                      return_coords=False,
                      ):
    """Get RA/Dec values, given an input structure

    Args:
        wcs: Input WCS
        jj: j coords
        ii: i coords
        units: Reference unit. Defaults to u.arcsec
        return_coords: If True, will return SkyCoords.
            Otherwise will convert
    """

    coords = wcs.pixel_to_world(jj, ii)

    if return_coords:
        return coords

    ra = coords.ra.to(units).value
    dec = coords.dec.to(units).value

    return ra, dec


def get_x_y_values(wcs,
                   ras,
                   decs,
                   units=u.arcsec,
                   ):
    """Get x/y values, given an input structure

    Args:
        wcs: Input WCS
        ras: RAs
        decs: Decs
        units: Reference unit. Defaults to u.arcsec
    """

    c = SkyCoord(ras * units, decs * units)
    x, y = wcs.world_to_pixel(c)

    return x, y


class LevelMatchStep:
    def __init__(
            self,
            in_dir,
            out_dir,
            step_ext,
            procs,
            band,
            fit_type_dithers="level",
            fit_type_recombine_lyot="level",
            fit_type_combine_nircam_short="level",
            fit_type_mosaic_tiles="level",
            recombine_lyot=False,
            combine_nircam_short=False,
            do_local_subtraction=True,
            sigma=3,
            npixels=3,
            dilate_size=7,
            max_iters=20,
            max_points=10000,
            do_sigma_clip=False,
            weight_method="equal",
            min_area_percent=0.002,
            min_linear_frac=0.25,
            rms_sig_limit=2,
            reproject_func="interp",
            overwrite=False,
    ):
        """Perform background matching between tiles

        This step performs background matching by minimizing the
        per-pixel differences between overlapping tiles. It does
        this first for dither groups, before creating a stacked
        image of these (to maximize areal coverage) and minimizing
        between all stacked images within a mosaic. This is necessary
        for observations that don't really have a background, and
        performs significantly better than the JWST implementation.

        N.B. If you use this, skymatch in the level 3 pipeline stage
        should be global or off, to avoid undoing this work

        Args:
            in_dir: Input directory
            out_dir: Output directory
            step_ext: .fits extension for the files going
                into the step
            procs: Number of parallel processes to run
            band: JWST band
            fit_type_dithers: What kind of fit to do to match levels between
                dithers in a single mosaic tile. Options are 'level' (the default,
                just fits a single offset), and 'level+slope' (which will fit a plane)
            fit_type_recombine_lyot: What kind of fit to do to match levels between
                science and lyot in a single mosaic tile. Options are 'level'
                (the default, just fits a single offset), and 'level+slope' (which will
                fit a plane)
            fit_type_combine_nircam_short: What kind of fit to do to match levels between
                the four NIRCam short chips. Options are 'level' (the default, just fits
                a single offset), and 'level+slope' (which will fit a plane)
            fit_type_mosaic_tiles: What kind of fit to do to match levels between
                mosaic tiles. Options are 'level' (the default, just fits a single offset),
                and 'level+slope' (which will fit a plane)
            recombine_lyot: If True, will recombine the lyot coronagraph
                into the main chip after the initial round of level matching.
                This will force the main science chip to have a 0 correction,
                as the lyot seems to be more wobbly. Defaults to False
            combine_nircam_short: Whether to combine the four NIRCam short
                chips before matching in a mosaic. Defaults to False
            do_local_subtraction: Whether to do a sigma-clipped local median
                subtraction. Defaults to True
            sigma: Sigma for sigma-clipping. Defaults to 3
            npixels: Pixels to grow for masking. Defaults to 5
            dilate_size: make_source_mask dilation size. Defaults to 7
            max_iters: Maximum sigma-clipping iterations. Defaults to 20
            max_points: Maximum points to include in histogram plots. This step can
                be slow so this speeds it up. Defaults to 10000
            do_sigma_clip: Whether to do sigma-clipping on data when reprojecting.
                Defaults to False
            weight_method: How to weight in least-squares minimization. Options are
                'equal' (equal weighting), 'npix' (weight by number of pixels), and
                'rms' (weight by rms of the delta values). Defaults to 'equal'
            min_area_percent: Minimum percentage of average areal overlap to remove tiles.
                Defaults to 0.002 (0.2%)
            min_linear_frac: Minimum linear overlap in any direction to keep tiles.
                Defaults to 0.25
            rms_sig_limit: Sigma limit for cutting off noisy fits. Defaults to 2
            reproject_func: Which reproject function to use. Defaults to 'interp',
                but can also be 'exact' or 'adaptive'
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        if reproject_func not in ALLOWED_REPROJECT_FUNCS:
            raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")
        if fit_type_dithers not in ALLOWED_FIT_TYPES:
            raise ValueError(f"fit_type_dithers should be one of {ALLOWED_FIT_TYPES}")
        if fit_type_recombine_lyot not in ALLOWED_FIT_TYPES:
            raise ValueError(f"fit_type_recombine_lyot should be one of {ALLOWED_FIT_TYPES}")
        if fit_type_combine_nircam_short not in ALLOWED_FIT_TYPES:
            raise ValueError(f"fit_type_combine_nircam_short should be one of {ALLOWED_FIT_TYPES}")
        if fit_type_mosaic_tiles not in ALLOWED_FIT_TYPES:
            raise ValueError(f"fit_type_mosaic_tiles should be one of {ALLOWED_FIT_TYPES}")

        if do_local_subtraction and fit_type_dithers not in ["level"]:
            log.warning("Cannot do local subtraction for methods beyond simple offset. Switching off")
            do_local_subtraction = False

        self.in_dir = in_dir
        self.out_dir = out_dir
        self.step_ext = step_ext
        self.procs = procs
        self.band = band
        self.fit_type_dithers = fit_type_dithers
        self.fit_type_recombine_lyot = fit_type_recombine_lyot
        self.fit_type_combine_nircam_short = fit_type_combine_nircam_short
        self.fit_type_mosaic_tiles = fit_type_mosaic_tiles
        self.recombine_lyot = recombine_lyot
        self.combine_nircam_short = combine_nircam_short
        self.do_local_subtraction = do_local_subtraction
        self.sigma = sigma
        self.npixels = npixels
        self.dilate_size = dilate_size
        self.max_iters = max_iters
        self.max_points = max_points
        self.do_sigma_clip = do_sigma_clip
        self.weight_method = weight_method
        self.min_area_percent = min_area_percent
        self.min_linear_frac = min_linear_frac
        self.rms_sig_limit = rms_sig_limit
        self.reproject_func = reproject_func
        self.overwrite = overwrite

        self.plot_dir = os.path.join(
            self.out_dir,
            "plots",
        )
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def do_step(self):
        """Run level matching"""

        if self.overwrite:
            shutil.rmtree(self.out_dir)

        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
            os.makedirs(self.plot_dir)

        # Check if we've already run the step
        step_complete_file = os.path.join(
            self.out_dir,
            "level_match_step_complete.txt",
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

        dithers = get_dithers(files,
                              )

        success = self.match_dithers(dithers,
                                     fit_type=self.fit_type_dithers,
                                     )
        if not success:
            log.warning("Failures detected level matching between individual dithers")
            return False

        if self.recombine_lyot:

            # First, check we're in MIRI mode
            band_type = get_band_type(self.band)

            if band_type in ["miri"]:

                success = self.match_lyot_science(dithers,
                                                  fit_type=self.fit_type_recombine_lyot,
                                                  )
                if not success:
                    log.warning("Failures detected level matching between lyot and main science")
                    return False

                # Redo the dithers, since now the "l" and "s" will have potentially been dropped
                files = glob.glob(
                    os.path.join(
                        self.out_dir,
                        f"*_{self.step_ext}.fits",
                    )
                )
                files.sort()

                # Split these into dithers per-chip
                dithers = get_dithers(files)

        if self.combine_nircam_short:

            # First, check we're in NIRCam short mode
            band_type, short_long_nircam = get_band_type(self.band,
                                                         short_long_nircam=True,
                                                         )

            if short_long_nircam in ["nircam_short"]:

                success = self.match_nircam_short(dithers,
                                                  fit_type=self.fit_type_combine_nircam_short,
                                                  )

                if not success:
                    log.warning("Failures detected level matching between individual dithers")
                    return False

                # Redo the dithers, since now we'll combine the 4 imaging chips for the short NIRCam
                files = glob.glob(
                    os.path.join(
                        self.out_dir,
                        f"*_{self.step_ext}.fits",
                    )
                )
                files.sort()

                # Split these into dithers per-chip
                dithers = get_dithers(files,
                                      combine_nircam_short=True,
                                      )

        if len(dithers) > 1:
            success = self.match_mosaic_tiles(dithers,
                                              fit_type=self.fit_type_mosaic_tiles,
                                              )
            if not success:
                log.warning("Failures detected level matching between mosaic tiles")
                return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def match_dithers(self,
                      dithers,
                      fit_type="level",
                      ):
        """Match levels between the dithers in each mosaic tile

        Args:
            dithers: List of dither groups
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
        """

        if fit_type not in ALLOWED_FIT_TYPES:
            raise ValueError(f"fit_type should be one of {ALLOWED_FIT_TYPES}")

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(dithers)])

        deltas, dither_files, ref_ras, ref_decs, optimal_wcses, optimal_shapes = self.get_per_dither_delta(
            dithers=dithers,
            fit_type=fit_type,
            procs=procs,
        )

        # Apply this calculated value
        for idx in range(len(deltas)):
            deltas_idx = copy.deepcopy(deltas[idx])
            dither_files_idx = copy.deepcopy(dither_files[idx])
            ref_ra = copy.deepcopy(ref_ras[idx])
            ref_dec = copy.deepcopy(ref_decs[idx])
            optimal_wcs = copy.deepcopy(optimal_wcses[idx])
            optimal_shape = copy.deepcopy(optimal_shapes[idx])

            # If we're including a local subtraction, do it here
            if self.do_local_subtraction:
                with datamodels.open(dither_files_idx[0]) as im:
                    data = copy.deepcopy(im.data)

                    # Mask out bad data
                    dq_bit_mask = get_dq_bit_mask(im.dq)
                    data[dq_bit_mask != 0] = np.nan
                    data[data == 0] = np.nan

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")

                        mask = make_source_mask(
                            data,
                            nsigma=self.sigma,
                            dilate_size=self.dilate_size,
                            sigclip_iters=self.max_iters,
                        )

                        # Calculate sigma-clipped median
                        local_delta = sigma_clipped_stats(
                            data,
                            mask=mask,
                            sigma=self.sigma,
                            maxiters=self.max_iters,
                        )[1]
                del im
            else:
                local_delta = 0

            for i, dither_file in enumerate(dither_files_idx):
                short_file = os.path.split(dither_file)[-1]
                out_file = os.path.join(
                    self.out_dir,
                    short_file,
                )
                delta = copy.deepcopy(deltas_idx[i])
                delta[-1] += local_delta

                print_delta(file=short_file,
                            delta=delta,
                            fit_type=fit_type,
                            )

                with datamodels.open(dither_file) as im:
                    im = apply_subtraction(im,
                                           delta,
                                           fit_type=fit_type,
                                           ref_ra=ref_ra,
                                           ref_dec=ref_dec,
                                           ref_wcs=optimal_wcs,
                                           ref_shape=optimal_shape,
                                           )
                    im.save(out_file)
                del im

        return True

    def match_lyot_science(self,
                           dithers,
                           fit_type="level",
                           ):
        """Match levels between each individual lyot/main science chip, and recombine

        Args:
            dithers: List of dithers
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
        """

        log.info("Matching levels between lyot and main chip and recombining")

        # From the individually corrected images
        # get a stacked image
        stacked_dir = self.out_dir + "_stacked"
        if not os.path.exists(stacked_dir):
            os.makedirs(stacked_dir)

        procs = np.nanmin([self.procs, len(dithers)])

        # We now want to find the separate l and s files
        combined_miri_dithers = []
        for dither in dithers:
            if dither[-1] in ["l", "s"]:
                combined_miri_dithers.append(dither[:-1])
        combined_miri_dithers = np.unique(combined_miri_dithers)

        if len(combined_miri_dithers) == 0:
            log.info("No split MIRI lyot/main chip found. Returning")
            return True

        successes = self.make_stacked_images(
            dithers=dithers,
            stacked_dir=stacked_dir,
            procs=procs,
        )

        if not np.all(successes):
            log.warning("Failures detected making stacked images")
            return False

        procs = np.nanmin([self.procs, len(combined_miri_dithers)])

        successes = []

        with mp.get_context("fork").Pool(procs) as pool:
            for success in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_match_lyot_science,
                            stacked_dir=stacked_dir,
                            fit_type=fit_type,
                        ),
                        combined_miri_dithers,
                    ),
                    total=len(combined_miri_dithers),
                    desc="Matching lyot and main science",
                    ascii=True,
                    disable=True,
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        # Remove the stacked images
        shutil.rmtree(stacked_dir)

        if not np.all(successes):
            log.warning("Failures detected matching lyot and main science")
            return False

        return True

    def match_nircam_short(self,
                           dithers,
                           fit_type="level",
                           ):
        """Match levels between the NIRCam short chips

        Args:
            dithers: List of dithers
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
        """

        log.info("Matching levels between NIRCam shorts")

        # From the individually corrected images
        # get a stacked image
        stacked_dir = self.out_dir + "_stacked"
        if not os.path.exists(stacked_dir):
            os.makedirs(stacked_dir)

        procs = np.nanmin([self.procs, len(dithers)])

        # We now want to group up the four NIRCam imaging chips
        combined_nircam_dithers = []
        for dither in dithers:
            combined_nircam_dithers.append(dither[:-1])
        combined_nircam_dithers = np.unique(combined_nircam_dithers)

        if len(combined_nircam_dithers) == 0:
            log.info("No NIRCam shorts found. Returning")
            return True

        successes = self.make_stacked_images(
            dithers=dithers,
            stacked_dir=stacked_dir,
            procs=procs,
        )

        if not np.all(successes):
            log.warning("Failures detected making stacked images")
            return False

        procs = np.nanmin([self.procs, len(combined_nircam_dithers)])

        successes = []

        with mp.get_context("fork").Pool(procs) as pool:
            for success in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_match_nircam_short,
                            stacked_dir=stacked_dir,
                            fit_type=fit_type,
                        ),
                        combined_nircam_dithers,
                    ),
                    total=len(combined_nircam_dithers),
                    desc="Matching NIRCAM short chips",
                    ascii=True,
                    disable=True,
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        # Remove the stacked images
        shutil.rmtree(stacked_dir)

        if not np.all(successes):
            log.warning("Failures detected matching NIRCam short chips")
            return False

        return True

    def match_mosaic_tiles(self,
                           dithers,
                           fit_type="level",
                           ):
        """Match levels between each mosaic tile

        Args:
            dithers: List of dither groups
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
        """

        log.info("Matching levels between mosaic tiles")

        # From the individually corrected images
        # get a stacked image
        stacked_dir = self.out_dir + "_stacked"
        if not os.path.exists(stacked_dir):
            os.makedirs(stacked_dir)

        procs = np.nanmin([self.procs, len(dithers)])

        successes = self.make_stacked_images(
            dithers=dithers,
            stacked_dir=stacked_dir,
            procs=procs,
        )

        if not np.all(successes):
            log.warning("Failures detected making stacked images")
            return False

        # Now match up these stacked images
        stacked_files = glob.glob(
            os.path.join(stacked_dir, f"*_{self.step_ext}.fits")
        )
        stacked_files.sort()

        # From these files, select a reference image based on the closest background,
        # if possible
        bgr_times = np.zeros(len(stacked_files))
        bgr_times[bgr_times == 0] = np.nan

        for i, f in enumerate(stacked_files):
            with fits.open(f) as hdu:
                if "DT_BGR" in hdu[0].header:
                    bgr_times[i] = float(hdu[0].header["DT_BGR"])

        if np.all(np.isnan(bgr_times)):
            # If we don't have any background times, let it select automatically
            ref_idx = 0
            log.info("Will use first image as the reference image")
        else:
            # Get the one closest in time to the backgrounds
            ref_idx = np.nanargmin(np.abs(bgr_times))

            # If it's selected a lyot image, force it to be the science
            tidy_file = os.path.split(stacked_files[ref_idx])[-1]
            full_file = stacked_files[ref_idx]

            if "mirimagel" in tidy_file:
                tidy_file = tidy_file.replace("mirimagel", "mirimages")
                full_file = full_file.replace("mirimagel", "mirimages")
                ref_idx = stacked_files.index(full_file)

            log.info(f"Will use {tidy_file} as the reference image")

        (
            delta_matrix,
            npix_matrix,
            rms_matrix,
            lin_size_matrix,
            valid_matrix,
            optimal_wcs,
            optimal_shape,
            ref_ra,
            ref_dec,
        ) = self.calculate_delta(
            stacked_files,
            fit_type=fit_type,
            stacked_image=True,
            procs=procs,
        )
        deltas = self.find_optimum_deltas(
            delta_mat=delta_matrix,
            npix_mat=npix_matrix,
            rms_mat=rms_matrix,
            lin_size_mat=lin_size_matrix,
            valid_mat=valid_matrix,
            fit_type=fit_type,
            ref_idx=ref_idx,
        )

        # Subtract the per-dither delta, do in place
        for idx, delta in enumerate(deltas):
            short_stack_file = os.path.split(stacked_files[idx])[-1]
            dither_files = glob.glob(
                os.path.join(
                    self.out_dir,
                    short_stack_file,
                )
            )
            dither_files.sort()

            print_delta(file=short_stack_file,
                        delta=delta,
                        fit_type=fit_type,
                        )

            for dither_file in dither_files:
                with datamodels.open(dither_file) as im:
                    im = apply_subtraction(im,
                                           delta,
                                           fit_type=fit_type,
                                           ref_ra=ref_ra,  # [idx],
                                           ref_dec=ref_dec,  # [idx],
                                           ref_wcs=optimal_wcs,
                                           ref_shape=optimal_shape,
                                           )
                    im.save(dither_file)
                del im

        # Remove the stacked images
        shutil.rmtree(stacked_dir)

        return True

    def get_per_dither_delta(
            self,
            dithers,
            fit_type="level",
            procs=1,
    ):
        """Function to parallelise getting the delta for each observation in a dither sequence

        Args:
            dithers: List of dithers to get deltas for
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
            procs: Number of processes to run simultaneously. Defaults
                to 1
        """

        log.info("Getting deltas for individual dithers")

        deltas = []
        dither_files = []
        ref_ras = []
        ref_decs = []
        optimal_wcses = []
        optimal_shapes = []

        with mp.get_context("fork").Pool(procs) as pool:
            for delta, dither_file, ref_ra, ref_dec, optimal_wcs, optimal_shape in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_per_dither_delta,
                            fit_type=fit_type,
                        ),
                        dithers,
                    ),
                    total=len(dithers),
                    desc="Matching individual dithers",
                    ascii=True,
            ):
                deltas.append(delta)
                dither_files.append(dither_file)
                ref_ras.append(ref_ra)
                ref_decs.append(ref_dec)
                optimal_wcses.append(optimal_wcs)
                optimal_shapes.append(optimal_shape)

            pool.close()
            pool.join()
            gc.collect()

        return deltas, dither_files, ref_ras, ref_decs, optimal_wcses, optimal_shapes

    def parallel_per_dither_delta(
            self,
            dither,
            fit_type="level",
    ):
        """Function to parallelise up matching dithers

        Args:
            dither: Input dither
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
        """

        dither_files = glob.glob(
            os.path.join(
                self.in_dir,
                f"{dither}*_{self.step_ext}.fits",
            )
        )
        dither_files.sort()

        with threadpool_limits(limits=1, user_api=None):
            (
                delta_matrix,
                npix_matrix,
                rms_matrix,
                lin_size_matrix,
                valid_matrix,
                optimal_wcs,
                optimal_shape,
                ref_ra,
                ref_dec,
            ) = self.calculate_delta(dither_files,
                                     fit_type=fit_type,
                                     )

        # If we're in a weird edge case where we only have one dither, jump out and just return 0s
        if len(dither_files) > 1:
            deltas = self.find_optimum_deltas(
                delta_mat=delta_matrix,
                npix_mat=npix_matrix,
                rms_mat=rms_matrix,
                lin_size_mat=lin_size_matrix,
                valid_mat=valid_matrix,
                fit_type=fit_type,
            )
        else:
            deltas = np.zeros([delta_matrix.shape[0], delta_matrix.shape[-1]])

        return deltas, dither_files, ref_ra, ref_dec, optimal_wcs, optimal_shape

    def make_stacked_images(
            self,
            dithers,
            stacked_dir,
            procs=1,
    ):
        """Function to parallellise up making stacked dither images

        Args:
            dithers: List of dithers to go
            stacked_dir: Where to save stacked images to
            procs: Number of simultaneous processes to run.
                Defaults to 1
        """

        log.info("Creating stacked images")

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in tqdm(
                    pool.imap_unordered(
                        partial(
                            self.parallel_make_stacked_image,
                            out_dir=stacked_dir,
                        ),
                        dithers,
                    ),
                    total=len(dithers),
                    desc="Creating stacked images",
                    ascii=True,
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes

    def parallel_make_stacked_image(
            self,
            dither,
            out_dir,
    ):
        """Light wrapper around parallelising the stacked image

        Args:
            dither: Dither to stack
            out_dir: Directory to save to
        """

        files = glob.glob(
            os.path.join(
                self.out_dir,
                f"{dither}*_{self.step_ext}.fits",
            )
        )
        files.sort()

        # Create output name
        file_name_split = os.path.split(files[0])[-1].split("_")
        file_name_split[2] = "*"
        out_name = "_".join(file_name_split)
        out_name = os.path.join(out_dir, out_name)

        # Make the stacked image. Set auto-rotate True to minimize the image shape
        success = make_stacked_image(
            files=files,
            out_name=out_name,
            additional_hdus="ERR",
            reproject_func=self.reproject_func,
            auto_rotate=True,
        )
        if not success:
            return False

        return True

    def parallel_match_lyot_science(self,
                                    dither,
                                    stacked_dir,
                                    fit_type
                                    ):
        """Function to parallelise up combining the lyot back into the main science chip

        Because the lyot seems to behave a little weirdly in its backgrounds from time-to-time,
        we force the main science correction to be 0 and put that all into the lyot

        Args:
            dither: Dither to level match and combine
            stacked_dir: Directory contained stacked images
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
        """

        stacked_files = glob.glob(
            os.path.join(stacked_dir, f"{dither}*_{self.step_ext}.fits")
        )
        stacked_files.sort()

        (
            delta_matrix,
            npix_matrix,
            rms_matrix,
            lin_size_matrix,
            valid_matrix,
            optimal_wcs,
            optimal_shape,
            ref_ra,
            ref_dec,
        ) = self.calculate_delta(
            stacked_files,
            fit_type=fit_type,
            stacked_image=True,
            procs=None,
        )
        lyot_delta, sci_delta = self.find_optimum_deltas(
            delta_mat=delta_matrix,
            npix_mat=npix_matrix,
            rms_mat=rms_matrix,
            lin_size_mat=lin_size_matrix,
            valid_mat=valid_matrix,
            fit_type=fit_type,
            ref_idx=1,  # Since we want to base the correction on the science files
        )

        # Subtract the per-dither delta. Since we've sorted, the first is lyot and the second is
        # the main science
        short_stack_file = os.path.split(stacked_files[1])[-1]
        sci_files = glob.glob(
            os.path.join(
                self.out_dir,
                short_stack_file,
            )
        )
        sci_files.sort()

        short_file = short_stack_file.replace("mirimages", "mirimage")

        print_delta(short_file,
                    lyot_delta,
                    fit_type=fit_type,
                    )

        for sci_file in sci_files:
            # Get the equivalent lyot file, and the out file (dropping that s/l)
            lyot_file = sci_file.replace("mirimages", "mirimagel")
            out_file = sci_file.replace("mirimages", "mirimage")

            with datamodels.open(sci_file) as sci_im, datamodels.open(lyot_file) as lyot_im:
                # Do the subtraction, only needed on the lyot since we force the science correction
                # to 0
                lyot_im = apply_subtraction(lyot_im,
                                            lyot_delta,
                                            fit_type=fit_type,
                                            ref_ra=ref_ra,
                                            ref_dec=ref_dec,
                                            ref_wcs=optimal_wcs,
                                            ref_shape=optimal_shape,
                                            )

                # Force the lyot back into the science
                sci_im.data[LYOT_I, LYOT_J] = lyot_im.data[LYOT_I, LYOT_J]
                sci_im.dq[LYOT_I, LYOT_J] = lyot_im.dq[LYOT_I, LYOT_J]

                # Save
                sci_im.save(out_file)

                del sci_im, lyot_im

                # And finally, remove the two separate files to clean things up
                os.system(f"rm -rf {sci_file}")
                os.system(f"rm -rf {lyot_file}")

        return True

    def parallel_match_nircam_short(self,
                                    dither,
                                    stacked_dir,
                                    fit_type
                                    ):
        """Function to parallelise up matching levels between the short NIRCam chips

        Args:
            dither: Dither to level match and combine
            stacked_dir: Directory contained stacked images
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
        """

        stacked_files = glob.glob(
            os.path.join(stacked_dir, f"{dither}*_{self.step_ext}.fits")
        )
        stacked_files.sort()

        (
            delta_matrix,
            npix_matrix,
            rms_matrix,
            lin_size_matrix,
            valid_matrix,
            optimal_wcs,
            optimal_shape,
            ref_ra,
            ref_dec,
        ) = self.calculate_delta(
            stacked_files,
            fit_type=fit_type,
            stacked_image=True,
            procs=None,
        )
        deltas = self.find_optimum_deltas(
            delta_mat=delta_matrix,
            npix_mat=npix_matrix,
            rms_mat=rms_matrix,
            lin_size_mat=lin_size_matrix,
            valid_mat=valid_matrix,
            fit_type=fit_type,
        )

        # Subtract the per-dither delta, do in place
        for idx, delta in enumerate(deltas):
            short_stack_file = os.path.split(stacked_files[idx])[-1]
            dither_files = glob.glob(
                os.path.join(
                    self.out_dir,
                    short_stack_file,
                )
            )
            dither_files.sort()

            print_delta(file=short_stack_file,
                        delta=delta,
                        fit_type=fit_type,
                        )

            for dither_file in dither_files:
                with datamodels.open(dither_file) as im:
                    im = apply_subtraction(im,
                                           delta,
                                           fit_type=fit_type,
                                           ref_ra=ref_ra,
                                           ref_dec=ref_dec,
                                           ref_wcs=optimal_wcs,
                                           ref_shape=optimal_shape,
                                           )
                    im.save(dither_file)
                del im

        return True

    def calculate_delta(
            self,
            files,
            fit_type="level",
            stacked_image=False,
            procs=None,
    ):
        """Match relative offsets between tiles

        Args:
            files (list): List of files to match
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
            stacked_image: Whether this is a stacked image or not.
                Default to False
            procs (int, optional): Number of processes to run in
                parallel. Defaults to None, which is series
        """

        if fit_type == "level":
            n_coeff = 1
        elif fit_type == "level+slope":
            n_coeff = 3
        else:
            raise ValueError(f"fit_type should be one of {ALLOWED_FIT_TYPES}")

        deltas = np.zeros([len(files), len(files), n_coeff])
        weights = np.zeros([len(files), len(files)])
        rmses = np.zeros_like(weights)
        lin_sizes = np.ones_like(weights)
        valid_mat = np.ones_like(weights)
        for i in range(len(files)):
            valid_mat[i, i] = 0

        # Reproject all the HDUs. Start by building the optimal WCS
        if isinstance(files[0], list):
            files_flat = list(itertools.chain(*files))
        else:
            files_flat = copy.deepcopy(files)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Get optimal WCS
            optimal_wcs, optimal_shape = find_optimal_celestial_wcs(
                files_flat,
                hdu_in="SCI",
                auto_rotate=True,
            )

        # The reference pixels are the centre of the shape, but we also want this as RA/Dec
        ref_y, ref_x = np.asarray(optimal_shape) // 2
        ref_ra, ref_dec = get_ra_dec_values(optimal_wcs, ref_x, ref_y)

        if procs is None:
            # Use a serial method

            # Reproject files, maintaining structure
            file_reproj = []

            for file in files:
                file_reproj.append(
                    self.get_reproject(
                        file=file,
                        optimal_wcs=optimal_wcs,
                        optimal_shape=optimal_shape,
                        stacked_image=stacked_image,
                    )
                )

            for i in range(len(files)):

                for j in range(i + 1, len(files)):
                    plot_name = self.get_plot_name(
                        files[i],
                        files[j],
                    )

                    n_pix, delta, rms, lin_size = self.get_level_match(
                        files1=file_reproj[i],
                        files2=file_reproj[j],
                        fit_type=fit_type,
                        ref_x=ref_x,
                        ref_y=ref_y,
                        plot_name=plot_name,
                    )

                    # These are symmetrical by design, but anything where we don't have values is invalid
                    if n_pix == 0 or delta is None or rms is None:
                        valid_mat[i, j] = 0
                        valid_mat[j, i] = 0
                        continue

                    for n in range(n_coeff):
                        deltas[j, i, n] = delta[n]
                        deltas[i, j, n] = -delta[n]

                    weights[j, i] = n_pix
                    rmses[j, i] = rms
                    lin_sizes[j, i] = lin_size

                    weights[i, j] = n_pix
                    rmses[i, j] = rms
                    lin_sizes[i, j] = lin_size

            gc.collect()

        else:
            # We can multiprocess this, since each calculation runs independently

            n_procs = np.nanmin([procs, len(files)])

            with mp.get_context("fork").Pool(n_procs) as pool:
                file_reproj = list([None] * len(files))

                for i, result in tqdm(
                        pool.imap_unordered(
                            partial(
                                self.parallel_get_reproject,
                                files=files,
                                optimal_wcs=optimal_wcs,
                                optimal_shape=optimal_shape,
                                stacked_image=stacked_image,
                            ),
                            range(len(files)),
                        ),
                        total=len(files),
                        desc="Reprojecting files",
                        ascii=True,
                ):
                    file_reproj[i] = result

                pool.close()
                pool.join()
                gc.collect()

            all_ijs = [
                (i, j) for i in range(len(files)) for j in range(i + 1, len(files))
            ]

            ijs = []
            delta_vals = []
            n_pix_vals = []
            rms_vals = []
            lin_size_vals = []

            for ij in tqdm(all_ijs, ascii=True, desc="Calculating delta matrix"):
                ij, delta, n_pix, rms, lin_size = self.parallel_delta_matrix(
                    ij=ij,
                    files=files,
                    fit_type=fit_type,
                    file_reproj=file_reproj,
                    ref_x=ref_x,
                    ref_y=ref_y,
                )

                ijs.append(ij)
                delta_vals.append(delta)
                n_pix_vals.append(n_pix)
                rms_vals.append(rms)
                lin_size_vals.append(lin_size)

            for idx, ij in enumerate(ijs):
                i = ij[0]
                j = ij[1]

                if n_pix_vals[idx] == 0 or delta_vals[idx] is None or rms_vals[idx] is None:
                    valid_mat[i, j] = 0
                    valid_mat[j, i] = 0
                    continue

                for n in range(n_coeff):
                    deltas[j, i, n] = delta_vals[idx][n]
                    deltas[i, j, n] = -delta_vals[idx][n]

                weights[j, i] = n_pix_vals[idx]
                rmses[j, i] = rms_vals[idx]
                lin_sizes[j, i] = lin_size_vals[idx]

                weights[i, j] = n_pix_vals[idx]
                rmses[i, j] = rms_vals[idx]
                lin_sizes[i, j] = lin_size_vals[idx]

            gc.collect()

        return deltas, weights, rmses, lin_sizes, valid_mat, optimal_wcs, optimal_shape, ref_ra, ref_dec

    def parallel_delta_matrix(
            self,
            ij,
            file_reproj,
            files,
            fit_type="level",
            ref_x=0,
            ref_y=0,
    ):
        """Function to parallelise up getting delta matrix values

        Args:
            ij: List of matrix (i, j) values
            file_reproj: Reprojected file
            files: Full list of files
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
            ref_x: Reference x value to subtract to centre the fits. Defaults to 0
            ref_y: Reference y value to subtract to centre the fits. Defaults to 0
        """

        i = ij[0]
        j = ij[1]

        plot_name = None
        if self.plot_dir is not None:
            plot_name = self.get_plot_name(
                files1=files[i],
                files2=files[j],
            )

        with threadpool_limits(limits=1, user_api=None):
            n_pix, delta, rms, lin_size = self.get_level_match(
                files1=file_reproj[i],
                files2=file_reproj[j],
                fit_type=fit_type,
                ref_x=ref_x,
                ref_y=ref_y,
                plot_name=plot_name,
            )

        gc.collect()

        return ij, delta, n_pix, rms, lin_size

    def get_level_match(
            self,
            files1,
            files2,
            fit_type,
            ref_x=0,
            ref_y=0,
            plot_name=None,
            maxiters=10,
            plane_fit_maxiters=20,
            plane_fit_abs_tol=0,
            plane_fit_rel_tol=1e-5,
    ):
        """Calculate relative difference between groups of files on the same pixel grid

        Args:
            files1: List of files to get difference from
            files2: List of files to get relative difference to
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
            ref_x: Reference x-coord to perform fits relative to. Defaults to 0
            ref_y: Reference y-coord to perform fits relative to. Defaults to 0
            plot_name: Output plot name. Defaults to None
            maxiters: Maximum iterations for the sigma-clipping. Defaults
                to 10
            plane_fit_maxiters: Maximum number of iterations for the plane fitting.
                Defaults to 20
            plane_fit_abs_tol: Absolute tolerance to define convergence in the
                plane fit. Defaults to 0 (i.e. don't use absolute tolerances)
            plane_fit_rel_tol: Relative tolerance to define convergence in
                the plane fitting. Defaults to 1e-5
        """

        diffs = []
        errs = []
        iis = []
        jjs = []

        if not isinstance(files1, list):
            files1 = [files1]
        if not isinstance(files2, list):
            files2 = [files2]

        n_pix = 0

        fig, axs = None, None
        if plot_name is not None:
            fig, axs = plt.subplots(
                nrows=len(files1),
                ncols=len(files2),
                figsize=(2.5 * len(files2), 2.5 * len(files1)),
                squeeze=False,
            )

        lin_size = 0

        for file_idx1, file1 in enumerate(files1):
            # Get out coordinates where data is valid, so we can do a linear
            # extent test
            file1_data = copy.deepcopy(file1["data"])
            file1_err = copy.deepcopy(file1["err"])

            ii, jj = np.indices(file1_data.array.shape, dtype=float)
            nan_idx = np.where(np.isnan(file1_data.array))

            ii[nan_idx] = np.nan
            jj[nan_idx] = np.nan

            # If we have something that's all NaNs
            # (e.g. lyot on MIRI subarray obs.), skip
            if np.all(np.isnan(ii)):
                continue

            file1_iaxis = np.nanmax(ii) - np.nanmin(ii)
            file1_jaxis = np.nanmax(jj) - np.nanmin(jj)

            for file_idx2, file2 in enumerate(files2):

                file2_data = copy.deepcopy(file2["data"])
                file2_err = copy.deepcopy(file2["err"])

                if file2_data.overlaps(file1_data):

                    # Get diffs, remove NaNs
                    diff = file2_data - file1_data
                    diff_arr = diff.array
                    diff_foot = diff.footprint
                    diff_arr[diff == 0] = np.nan
                    diff_arr[diff_foot == 0] = np.nan

                    # Pull out error arrays, remove NaNs
                    err = file1_err * file1_err + file2_err * file2_err
                    err_arr = np.sqrt(err.array)
                    err_arr[diff == 0] = np.nan
                    err_arr[diff_foot == 0] = np.nan

                    # Get out coordinates where data is valid, so we can do a linear
                    # extent test
                    ii, jj = np.indices(file2_data.array.shape, dtype=float)
                    ii[np.where(np.isnan(file2_data.array))] = np.nan
                    jj[np.where(np.isnan(file2_data.array))] = np.nan

                    # If we have something that's all NaNs
                    # (e.g. lyot on MIRI subarray obs.), skip
                    if np.all(np.isnan(ii)):
                        continue

                    file2_iaxis = np.nanmax(ii) - np.nanmin(ii)
                    file2_jaxis = np.nanmax(jj) - np.nanmin(jj)

                    ii, jj = np.indices(diff_arr.shape, dtype=float)
                    ii[np.where(np.isnan(diff_arr))] = np.nan
                    jj[np.where(np.isnan(diff_arr))] = np.nan

                    # If the slices are all NaNs, then we can just move on
                    if np.all(np.isnan(ii)):
                        continue

                    diff_iaxis = np.nanmax(ii) - np.nanmin(ii)
                    diff_jaxis = np.nanmax(jj) - np.nanmin(jj)

                    # Include everything, but flag if we've hit the minimum linear extent
                    if (
                            diff_iaxis > file1_iaxis * self.min_linear_frac
                            or diff_jaxis > file1_jaxis * self.min_linear_frac
                            or diff_iaxis > file2_iaxis * self.min_linear_frac
                            or diff_jaxis > file2_jaxis * self.min_linear_frac
                    ):
                        lin_size = 1

                    valid_idx = np.isfinite(diff_arr)

                    # Get the coords, account for the differences in where the arrays start
                    # IMPORTANT: Our indexing conventions are different to reproject, so i/j
                    # gets swapped
                    diff_ii = ii[valid_idx] + diff.jmin
                    diff_jj = jj[valid_idx] + diff.imin

                    diff = diff_arr[valid_idx].tolist()
                    err = err_arr[valid_idx].tolist()
                    n_pix += len(diff)

                    diffs.extend(diff)
                    errs.extend(err)
                    iis.extend(diff_ii)
                    jjs.extend(diff_jj)

                    if plot_name is not None:
                        if len(diff) > 0:
                            vmin, vmax = np.nanpercentile(diff, [1, 99])
                            axs[file_idx1, file_idx2].imshow(
                                diff_arr,
                                origin="lower",
                                vmin=vmin,
                                vmax=vmax,
                                interpolation="nearest",
                            )
                if plot_name is not None:
                    axs[file_idx1, file_idx2].set_axis_off()

        if plot_name is not None:
            if n_pix > 0:
                plt.savefig(f"{plot_name}_ims.png", bbox_inches="tight")
                plt.savefig(f"{plot_name}_ims.pdf", bbox_inches="tight")
            plt.close()

        if n_pix > 0:

            if fit_type == "level":
                # Just fit a DC offset. Sigma-clip to remove outliers in the distribution
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, delta, rms = sigma_clipped_stats(diffs, sigma=self.sigma, maxiters=maxiters)

                if plot_name is not None:
                    # Get histogram range
                    diffs_hist = None

                    if self.max_points is not None:
                        if len(diffs) > self.max_points:
                            diffs_hist = random.sample(diffs, self.max_points)
                    if diffs_hist is None:
                        diffs_hist = copy.deepcopy(diffs)

                    hist_range = np.nanpercentile(diffs_hist, [1, 99])

                    plt.figure(figsize=(5, 4))
                    plt.hist(
                        diffs_hist,
                        histtype="step",
                        bins=50,
                        range=hist_range,
                        color="gray",
                    )
                    plt.axvline(
                        delta,
                        c="k",
                        ls="--",
                    )

                    plt.xlabel("Diff (MJy/sr)")
                    plt.ylabel("$N$")

                    plt.grid()

                    plt.tight_layout()

                    plt.savefig(f"{plot_name}_hist.pdf", bbox_inches="tight")
                    plt.savefig(f"{plot_name}_hist.png", bbox_inches="tight")
                    plt.close()

                # For consistency with how we'll do things later, put this into a list
                delta = [delta]

            elif fit_type == "level+slope":

                diffs = np.array(diffs)
                errs = np.array(errs)

                # For this, we do a slope fit in the x, y, rather than RA/Dec plane
                # to avoid any potential spherical weirdness
                iis = np.array(iis)
                jjs = np.array(jjs)

                jjs -= ref_x
                iis -= ref_y

                # Also remove a mean x/y, to avoid fitting spurious correlations
                mean_j = np.nanmean(jjs)
                mean_i = np.nanmean(iis)
                jjs -= mean_j
                iis -= mean_i

                initial_offset = np.nanmedian(diffs)

                # Get an initial guess which we'll normalise things by. This is just a flat plane
                delta = np.array([0, 0, initial_offset], dtype=float)

                # Set up an initial difference for comparison, with really big numbers
                delta_diff = np.ones_like(delta) * 1e9

                converged = False
                n_iter = 0

                while not converged and n_iter <= plane_fit_maxiters:

                    # prev_delta = copy.deepcopy(delta)
                    prev_delta_diff = copy.deepcopy(delta_diff)

                    # Look at the current plane we have, and reject points that are
                    # significantly different to it
                    delta_plane = plane(jjs, iis, delta)
                    rms = np.nanstd(diffs - delta_plane)

                    fit_idx = np.where(np.abs(diffs - delta_plane) < self.sigma * rms)

                    # Do a fit to the residuals, using only the points we care about
                    points = np.vstack(
                        (
                            jjs[fit_idx],
                            iis[fit_idx],
                            diffs[fit_idx] - delta_plane[fit_idx],
                        )
                    ).T

                    func = partial(plane_resid,
                                   points=points,
                                   err=errs[fit_idx],
                                   )
                    res = minimize(func,
                                   delta,
                                   method="Powell",
                                   )

                    delta_diff = copy.deepcopy(res.x)

                    # Add this to our final delta
                    delta += delta_diff

                    # If the changes are very small, then just call this converged and jump out
                    if np.all(np.isclose(delta_diff,
                                         prev_delta_diff,
                                         atol=plane_fit_abs_tol,
                                         rtol=plane_fit_rel_tol,
                                         )
                              ):
                        converged = True
                        log.debug(f"Plane fitting converged after {n_iter} iterations")

                    n_iter += 1

                if not converged:
                    log.debug(f"Plane fitting did not converge after {n_iter-1} iterations")

                ii_min, ii_max = np.nanmin(iis), np.nanmax(iis)
                jj_min, jj_max = np.nanmin(jjs), np.nanmax(jjs)

                # Get residuals from the best plane
                best_plane = plane(
                    jjs,
                    iis,
                    delta,
                )
                resid = diffs - best_plane

                # Get a measure of the RMS
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, _, rms = sigma_clipped_stats(resid,
                                                    sigma=self.sigma,
                                                    maxiters=maxiters,
                                                    )

                # Get histogram range for residuals
                resid_hist = None

                if self.max_points is not None:
                    if len(resid) > self.max_points:
                        resid_hist = random.sample(list(resid), self.max_points)
                if resid_hist is None:
                    resid_hist = copy.deepcopy(resid)

                # And also points to scatter
                scatter_idx = None

                if self.max_points is not None:
                    if len(resid) > self.max_points:
                        scatter_idx = np.array(random.sample(range(len(resid)), self.max_points))
                if scatter_idx is None:
                    scatter_idx = slice(None)

                hist_range = np.nanpercentile(resid, [1, 99])

                # Get the plane to show the fit
                xx, yy = np.meshgrid([jj_min, jj_max], [ii_min, ii_max])
                z = delta[0] * xx + delta[1] * yy + delta[2]

                # Make a plot that shows the plane fit through the points on the left, and the
                # residuals on the right
                fig = plt.figure(figsize=(8, 4))

                ax_3d = fig.add_subplot(1, 2, 1, projection='3d')

                ax_3d.set_proj_type('ortho')
                ax_3d.view_init(elev=30, azim=45, roll=0)

                ax_3d.scatter(
                    jjs[scatter_idx],
                    iis[scatter_idx],
                    diffs[scatter_idx],
                    c='k',
                    marker='.',
                    alpha=0.1,
                    rasterized=True,
                )
                ax_3d.plot_surface(xx, yy, z, alpha=0.4, color='red')

                ax_3d.set_xlim(jj_min, jj_max)
                ax_3d.set_ylim(ii_min, ii_max)
                ax_3d.ticklabel_format(useOffset=False)

                if ref_x is not None:
                    x_label = r"$\Delta x$"
                else:
                    x_label = r"$x$"

                if ref_y is not None:
                    y_label = r"$\Delta y$"
                else:
                    y_label = r"$y$"

                ax_3d.set_xlabel(f"{x_label} (pix)")
                ax_3d.set_ylabel(f"{y_label} (pix)")
                ax_3d.set_zlabel("Diff (MJy/sr)")

                ax_3d.xaxis.labelpad = 10
                ax_3d.yaxis.labelpad = 10

                ax_resid = fig.add_subplot(1, 2, 2)
                ax_resid.hist(
                    resid_hist,
                    histtype="step",
                    bins=50,
                    range=hist_range,
                    color="gray",
                )
                ax_resid.set_xlabel("Residual (MJy/sr)")
                ax_resid.set_ylabel("$N$")

                ax_resid.yaxis.set_label_position("right")
                ax_resid.yaxis.tick_right()

                ax_resid.grid()

                plt.tight_layout()

                plt.savefig(f"{plot_name}_plane_fit.pdf", bbox_inches="tight", dpi=300)
                plt.savefig(f"{plot_name}_plane_fit.png", bbox_inches="tight", dpi=300)
                plt.close()

                # Translate this back to the central coordinate, since we subtracted that off earlier
                z_offset = delta[0] * mean_j + delta[1] * mean_i
                delta[-1] -= z_offset

            else:
                raise ValueError(f"fit_type should be one of {ALLOWED_FIT_TYPES}")

        else:
            delta = None
            rms = None

        gc.collect()

        return n_pix, delta, rms, lin_size

    def find_optimum_deltas(
            self,
            delta_mat,
            npix_mat,
            rms_mat,
            lin_size_mat,
            valid_mat,
            fit_type="level",
            n_draws=25,
            n_iter=10000,
            convergence_abs_tol=0,
            convergence_rel_tol=1e-5,
            ref_idx=None,
    ):
        """Get optimum deltas from a delta/weight matrix.

        Taken from the JWST skymatch step, with some edits to remove potentially bad fits due
        to small areal overlaps, or noisy diffs, and various weighting schemes.

        If we're fitting a plane, delta_mat will be an NxNx3 matrix, and we'll minimize over
        each of the last axes separately

        Args:
            delta_mat (np.ndarray): Matrix of delta values. These may be [a, b, c] coefficients
                if we're fitting a plane
            npix_mat (np.ndarray): Matrix of number of pixel values for calculating delta
            rms_mat (np.ndarray): Matrix of RMS values
            lin_size_mat (np.ndarray): 1/0 array for whether overlaps pass minimum linear extent
            valid_mat (np.ndarray): 1/0 array for whether overlaps are valid
            fit_type: Which type of fit to do. See ALLOWED_FIT_TYPES. Defaults to "level"
            n_draws: When using the iterative method, we need to sample from the fitted plane. This
                controls how many draws we do. Defaults to 25
            n_iter: Maximum number of iterations before breaking out of the fitting routine. Defaults
                to 10,000
            convergence_abs_tol: Absolute tolerance to define convergence. Defaults to 0 (i.e. don't use
                absolute tolerances)
            convergence_rel_tol: Relative tolerance to define convergence. Defaults to 1e-5
            ref_idx: Index to define the zero level for all the level matching. Defaults to None, which
                will use the average correction
        """

        delta_mat = copy.deepcopy(delta_mat)
        npix_mat = copy.deepcopy(npix_mat)
        rms_mat = copy.deepcopy(rms_mat)
        lin_size_mat = copy.deepcopy(lin_size_mat)

        ns = delta_mat.shape[0]

        # Matrix for fits that we'll actually use
        use_mat = copy.deepcopy(valid_mat)

        # Remove things with weights less than min_area_percent of the average weight. Use all overlaps here
        avg_npix_val = np.nanmean(npix_mat[valid_mat == 1])
        small_area_idx = npix_mat < self.min_area_percent * avg_npix_val

        use_mat[small_area_idx] = 0

        # Remove things that haven't passed the small area test
        use_mat[lin_size_mat == 0] = 0

        # Remove fits with RMS values some sigma above the mean. Use only good overlaps here
        avg_rms_val = np.nanmean(rms_mat[np.logical_and(valid_mat == 1, use_mat == 1)])
        sig_rms_val = np.nanstd(rms_mat[np.logical_and(valid_mat == 1, use_mat == 1)])

        rms_idx = np.where(rms_mat > avg_rms_val + self.rms_sig_limit * sig_rms_val)

        use_mat[rms_idx] = 0

        # Create weight matrix
        if self.weight_method == "equal":
            # Weight evenly
            weight = np.ones_like(delta_mat)
        elif self.weight_method == "npix":
            # Weight by straight number of pixels
            weight = 0.5 * (npix_mat + npix_mat.T)
        elif self.weight_method == "rms":
            # Weight by inverse variance of the fit
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                weight = 0.5 * (rms_mat + rms_mat.T)
                weight = weight ** -2
                weight[~np.isfinite(weight)] = 0
        else:
            raise ValueError(f"weight_method {self.weight_method} not known")

        neq = 0
        for i in range(ns):
            for j in range(i + 1, ns):
                if valid_mat[i, j] == 1 and use_mat[i, j] == 1:
                    neq += 1

        # Create arrays for coefficients and free terms
        k = np.zeros((neq, ns), dtype=float)
        f = np.zeros([neq, delta_mat.shape[-1]], dtype=float)
        invalid = ns * [True]

        # Process intersections between the rest of the images
        ieq = 0
        for i in range(0, ns):
            for j in range(i + 1, ns):
                # Only pull out valid intersections
                if valid_mat[i, j] == 1 and use_mat[i, j] == 1:
                    k[ieq, i] = weight[i, j]
                    k[ieq, j] = -weight[i, j]

                    for coeff in range(delta_mat.shape[-1]):
                        f[ieq, coeff] = weight[i, j] * delta_mat[i, j, coeff]

                    invalid[i] = False
                    invalid[j] = False
                    ieq += 1

        rank = np.linalg.matrix_rank(k, 1.0e-12)

        if rank < ns - 1:
            logging.warning(
                f"There are more unknown sky values ({ns}) to be solved for"
            )
            logging.warning(
                "than there are independent equations available "
                f"(matrix rank={rank})."
            )
            logging.warning("Sky matching (delta) values will be computed only for")
            logging.warning("a subset (or more independent subsets) of input images.")

        # Uses the iterative montage method to find best fits

        deltas = np.zeros([ns, delta_mat.shape[-1]])
        delta_mat_corr = copy.deepcopy(delta_mat)

        converged = False
        level_converged = False

        # Set a max number of iterations to just run level matching. This is
        # either 2500 or half the number of iterations if the iteration
        # number is relatively small
        if n_iter < 5000:
            n_level = n_iter // 2
        else:
            n_level = 2500

        iteration = 0
        level_iteration = 0

        # Iterate until convergence or some maximum number of iterations
        while not converged and iteration < n_iter:

            # We'll start off just trying to optimize the DC offsets
            if fit_type in ["level+slope"] and not level_converged:
                curr_fit_type = "level"
            else:
                curr_fit_type = copy.deepcopy(fit_type)

            # Pull useful things out to dictionaries since we'll update
            # everything in bulk at the end
            delta_arr = {}
            best_fits = {}

            delta_mat_corr_prev = copy.deepcopy(delta_mat_corr)

            for i in range(ns):
                if invalid[i]:
                    continue

                delta_arr[i] = {}

                # Loop over and pull out any valid overlaps
                for j in range(ns):
                    if i == j:
                        continue

                    if invalid[j]:
                        continue

                    # Only take fits we actually want to use
                    if valid_mat[i, j] == 1 and use_mat[i, j] == 1:
                        delta_arr[i][j] = copy.deepcopy(delta_mat_corr[i, j, :])

                # For each delta value, we want to sample randomly in the (x, y) plane
                weight_vals = []
                x_vals = []
                y_vals = []
                z_vals = []

                # For a starting guess, take the average of the various coefficients
                delta_arr_stack = np.array([list(delta_arr[i][d]) for d in delta_arr[i]])
                p0 = np.nanmean(delta_arr_stack, axis=0)

                # Sample some points in the plane
                x = np.random.normal(loc=0, scale=1, size=n_draws)
                y = np.random.normal(loc=0, scale=1, size=n_draws)

                for j in delta_arr[i]:

                    if invalid[j]:
                        continue

                    # This is where we're fitting, so only use the values that we've defined as good
                    if valid_mat[i, j] == 0 or use_mat[i, j] == 0:
                        continue

                    arr_val = copy.deepcopy(delta_arr[i][j])

                    if curr_fit_type == "level":
                        coeffs = np.array([0, 0, arr_val[-1]])
                    elif curr_fit_type == "level+slope":
                        coeffs = copy.deepcopy(arr_val)
                    else:
                        raise ValueError(f"Unknown fit type: {curr_fit_type}")

                    z = coeffs[0] * x + coeffs[1] * y + coeffs[2]

                    weight_vals.extend([weight[i, j]] * n_draws)
                    x_vals.extend(list(x))
                    y_vals.extend(list(y))
                    z_vals.extend(list(z))

                x_vals = np.array(x_vals)
                y_vals = np.array(y_vals)
                z_vals = np.array(z_vals)
                weight_vals = np.array(weight_vals)

                if curr_fit_type == "level":

                    # Just get an average of the offsets
                    best_fit_vals = np.average(z_vals,
                                               weights=np.sqrt(weight_vals),
                                               keepdims=True,
                                               )

                elif curr_fit_type == "level+slope":

                    # Use scipy minimize to get a best fit plane
                    points = np.vstack((x_vals, y_vals, z_vals)).T

                    # Here, the error is the inverse of the weights
                    func = partial(plane_resid,
                                   points=points,
                                   err=np.sqrt(weight_vals) ** -1,
                                   )
                    res = minimize(func,
                                   p0,
                                   method="Powell",
                                   )

                    # Pull out best fit, calculate the stats and update
                    # which points we're fitting
                    best_fit_vals = copy.deepcopy(res.x)

                else:
                    raise ValueError(f"Unknown fit type: {curr_fit_type}")

                if curr_fit_type == "level":
                    best_fit = np.zeros(delta_mat.shape[-1])
                    best_fit[-1] = best_fit_vals[-1]
                elif curr_fit_type == "level+slope":
                    best_fit = copy.deepcopy(best_fit_vals)
                else:
                    raise ValueError(f"Fit type {curr_fit_type} is not known.")

                # Factor of 2 to keep things symmetrical
                best_fit /= 2

                best_fits[i] = copy.deepcopy(best_fit)

            for i in range(ns):
                if i not in best_fits:
                    continue

                # Add on the best fits
                deltas[i, :] += best_fits[i]

            # Edit the corrections, doing this for all valid fits
            for i in range(ns):

                if invalid[i]:
                    continue

                if i not in best_fits:
                    continue

                for j in range(ns):

                    if i == j:
                        continue

                    if invalid[j]:
                        continue

                    if valid_mat[i, j] == 0:
                        continue

                    # Apply the corrections within the matrix
                    delta_mat_corr[i, j, :] -= best_fits[i]
                    delta_mat_corr[i, j, :] += best_fits[j]

            # Check for convergence. If the maximum difference hasn't
            # changed within the tolerance, jump out and call it a day
            has_converged = np.all(np.isclose(delta_mat_corr,
                                              delta_mat_corr_prev,
                                              atol=convergence_abs_tol,
                                              rtol=convergence_rel_tol,
                                              )
                                   )

            # if convergence_param < convergence_tol:
            if has_converged:

                if curr_fit_type != fit_type:
                    level_converged = True
                else:
                    converged = True
                log.info(f"Level matching converged after {iteration} iterations")

            if level_iteration >= n_level and not level_converged:
                log.info(f"Level matching has not converged after {iteration} iterations")
                level_converged = True

            # Update convergences
            if not level_converged:
                level_iteration += 1
            if not converged:
                iteration += 1

        # If we don't have a selected reference index, take the average correction
        if ref_idx is None:
            offset_delta = np.nanmean(deltas, axis=0)
        else:
            offset_delta = copy.deepcopy(deltas[ref_idx, :])

        # Set the reference image correction to 0, adjust all other
        # corrections relative to that
        for i in range(deltas.shape[0]):
            deltas[i, :] -= offset_delta

        # Set any invalid deltas to 0
        deltas[np.asarray(invalid, dtype=bool), :] = 0

        return deltas

    def get_reproject(
            self,
            file,
            optimal_wcs,
            optimal_shape,
            stacked_image=False,
    ):
        """Reproject files, maintaining list structure

        Args:
            file: List or single file to reproject
            optimal_wcs: WCS to reproject to
            optimal_shape: output array shape for the WCS
            stacked_image (bool): Whether this is a stacked image or not. Defaults to False
        """

        if isinstance(file, list):
            file_reproj = [
                {"data": reproject_image(
                    i,
                    optimal_wcs=optimal_wcs,
                    optimal_shape=optimal_shape,
                    do_sigma_clip=self.do_sigma_clip,
                    stacked_image=stacked_image,
                    reproject_func=self.reproject_func,
                ),
                    "err": reproject_image(
                        i,
                        optimal_wcs=optimal_wcs,
                        optimal_shape=optimal_shape,
                        hdu_type="err",
                        do_sigma_clip=self.do_sigma_clip,
                        stacked_image=stacked_image,
                        reproject_func=self.reproject_func,
                    ),
                }
                for i in file
            ]
        else:
            file_reproj = {
                "data": reproject_image(
                    file,
                    optimal_wcs=optimal_wcs,
                    optimal_shape=optimal_shape,
                    do_sigma_clip=self.do_sigma_clip,
                    stacked_image=stacked_image,
                    reproject_func=self.reproject_func,
                ),
                "err": reproject_image(
                    file,
                    optimal_wcs=optimal_wcs,
                    optimal_shape=optimal_shape,
                    hdu_type="err",
                    do_sigma_clip=self.do_sigma_clip,
                    stacked_image=stacked_image,
                    reproject_func=self.reproject_func,
                ),
            }

        return file_reproj

    def parallel_get_reproject(
            self,
            idx,
            files,
            optimal_wcs,
            optimal_shape,
            stacked_image=False,
    ):
        """Light function to parallelise get_dither_reproject

        Args:
            idx: File idx to reproject
            files: Full file list
            optimal_wcs: Optimal WCS for input stack of images
            optimal_shape: Optimal shape for input stack of images
            stacked_image: Stacked image or not? Defaults to False
        """

        dither_reproj = self.get_reproject(
            file=files[idx],
            optimal_wcs=optimal_wcs,
            optimal_shape=optimal_shape,
            stacked_image=stacked_image,
        )

        return idx, dither_reproj

    def get_plot_name(
            self,
            files1,
            files2,
    ):
        """Make a plot name from list of files for level matching

        Args:
            files1: First list of files
            files2: Second list of files
        """
        if isinstance(files1, list):
            files1_name_split = os.path.split(files1[0])[-1].split("_")

            # Since these should be dither groups, blank out the dither
            files1_name_split[2] = "XXXXX"
        else:
            files1_name_split = os.path.split(files1)[-1].split("_")
        plot_to_name = "_".join(files1_name_split[:-1])

        if isinstance(files2, list):
            files2_name_split = os.path.split(files2[0])[-1].split("_")

            # Since these should be dither groups, blank out the dither
            files2_name_split[2] = "XXXXX"
        else:
            files2_name_split = os.path.split(files2)[-1].split("_")
        plot_from_name = "_".join(files2_name_split[:-1])

        plot_name = os.path.join(
            self.plot_dir,
            f"{plot_from_name}_to_{plot_to_name}_level_match",
        )

        return plot_name
