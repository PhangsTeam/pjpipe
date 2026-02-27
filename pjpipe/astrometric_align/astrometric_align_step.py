import copy
import gc
import glob
import logging
import multiprocessing as mp
import os
import warnings
from functools import partial

import gwcs
from astropy.io import fits

try:
    from gwcs.utils import make_fitswcs_transform
except ImportError:
    from gwcs.wcs.utils import make_fitswcs_transform

import numpy as np
from astropy.table import QTable, Table
from astropy.wcs import WCS
from image_registration import cross_correlation_shifts
from jwst.assign_wcs.util import update_fits_wcsinfo
from reproject import reproject_interp, reproject_adaptive, reproject_exact
from stdatamodels.jwst import datamodels
from tqdm import tqdm
from tweakwcs import fit_wcs, XYXYMatch
from tweakwcs.correctors import FITSWCSCorrector, JWSTWCSCorrector

from ..utils import (
    get_band_type,
    parse_parameter_dict,
    recursive_setattr,
    get_default_args,
    get_kws,
)

ALLOWED_REPROJECT_FUNCS = [
    "interp",
    "adaptive",
    "exact",
]

log = logging.getLogger(__name__)


def get_lv3_wcs(im):
    """Get a useful WCS from a JWST mosaic

    Args:
        im: JWST datamodel
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        fits_hdr = im.meta.wcs.to_fits()[0]
        wcs_hdr = im.meta.wcsinfo.instance
        naxis1, naxis2 = fits_hdr["NAXIS1"], fits_hdr["NAXIS2"]
        wcs_hdr["naxis1"] = naxis1
        wcs_hdr["naxis2"] = naxis2

        wcs_im = WCS(wcs_hdr)

    return wcs_im


def transform_wcs_gwcs(wcs):
    """Convert WCS to gWCS

    Args:
        wcs: Astropy WCS instance
    """

    hdr = wcs.to_header()
    tform = make_fitswcs_transform(hdr)
    new_gwcs = gwcs.WCS(forward_transform=tform, output_frame="world")

    return hdr, new_gwcs


def lv3_update_fits_wcsinfo(im, hdr):
    """Quick wrapper to fix up level 3 datamodel wcsinfo

    Args:
        im: JWST datamodel
        hdr: Header instance
    """

    # update meta.wcsinfo with FITS keywords except for naxis*
    del hdr["naxis*"]

    # maintain convention of lowercase keys
    hdr_dict = {k.lower(): v for k, v in hdr.items()}

    # delete naxis, cdelt, pc from wcsinfo
    rm_keys = [
        "naxis",
        "cdelt1",
        "cdelt2",
        "pc1_1",
        "pc1_2",
        "pc2_1",
        "pc2_2",
        "a_order",
        "b_order",
        "ap_order",
        "bp_order",
    ]

    rm_keys.extend(
        f"{s}_{i}_{j}"
        for i in range(10)
        for j in range(10)
        for s in ["a", "b", "ap", "bp"]
    )

    for key in rm_keys:
        if key in im.meta.wcsinfo.instance:
            del im.meta.wcsinfo.instance[key]

    # update meta.wcs_info with fit keywords
    im.meta.wcsinfo.instance.update(hdr_dict)

    return im


def parallel_tweakback(
        f,
        matrix=None,
        shift=None,
        ref_tpwcs=None,
):
    """Wrapper function to parallelise tweakback routine

    Args:
        f: File to tweakback
        matrix: rotation/skew matrix. Defaults to None
        shift: [x, y] shift. Defaults to None
        ref_tpwcs: WCS in which shift is defined. Defaults
            to None
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if matrix is None:
            matrix = [[1, 0], [0, 1]]
        if shift is None:
            shift = [0, 0]

        input_im = datamodels.open(f)

        input_wcs = input_im.meta.wcs
        input_wcsinfo = input_im.meta.wcsinfo.instance

        im = JWSTWCSCorrector(
            wcs=input_wcs,
            wcsinfo=input_wcsinfo,
        )
        im.set_correction(
            matrix=matrix,
            shift=shift,
            ref_tpwcs=ref_tpwcs,
        )

        input_im.meta.wcs = im.wcs

        try:
            update_fits_wcsinfo(
                input_im,
            )
        except (ValueError, RuntimeError) as e:
            logging.warning(
                "Failed to update 'meta.wcsinfo' with FITS SIP "
                f'approximation. Reported error is:\n"{e.args[0]}"'
            )
            return False

    out_file = f.replace(".fits", "_tweakback.fits")
    input_im.save(out_file)

    del im
    del input_im
    gc.collect()

    return True


class AstrometricAlignStep:
    def __init__(
            self,
            target,
            bands,
            progress_dict,
            target_dir,
            catalog_dir,
            step_ext,
            procs,
            step_parameters,
            catalogs=None,
            align_mapping_mode="shift",
            align_mapping=None,
            tweakreg_parameters=None,
            reproject_func="interp",
            overwrite=False,
    ):
        """Perform absolute astrometric alignment

        There are a number of modes here. The simplest
        is by matching a catalog of sources, using
        tweakreg. Alternatively, we can either apply
        calculated shifts to other mosaics, or attempt
        to match via cross-correlation.

        Args:
            target: Target to consider
            bands: Bands to consider
            catalog_dir: Directory of alignment catalogs
            step_ext: .fits extension for the files going
                into the step
            procs: Number of processes to run in parallel
            catalogs: Dictionary for the external alignment
                catalogs
            align_mapping_mode: If locking to other JWST image,
                method to use. Option is "shift" (pull the
                tweakreg solution from the existing file),
                or "cross-corr" (do some cross-correlation
                between the images)
            tweakreg_parameters: Dictionary of parameters
                to pass to tweakreg for the standard alignment
            reproject_func: Which reproject function to use. Defaults to 'interp',
                but can also be 'exact' or 'adaptive'
            overwrite: Whether to overwrite or not. Defaults
                to False
        """

        self.target = target
        self.bands = bands
        self.progress_dict = progress_dict
        self.target_dir = target_dir
        self.catalog_dir = catalog_dir
        self.step_ext = step_ext
        self.procs = procs
        self.step_parameters = step_parameters

    def do_step(self):
        """Run absolute astrometric alignment"""

        # Pull out to a band order where we do the reference bands first

        align_mappings = self.step_parameters.get("align_mapping", {})
        reference_bands = np.unique([align_mappings[k] for k in align_mappings])
        reference_bands = [str(x) for x in reference_bands]

        non_reference_bands = [
            b for b in self.bands
            if b not in reference_bands
        ]

        bands = reference_bands + non_reference_bands

        successes = []
        for band in bands:
            success = self.do_step_band(band)
            successes.append(success)

        if not all(successes):
            return False

        return True

    def do_step_band(self, band):
        """Run absolute astrometric alignment per-band

        Args:
            band: Band to consider
        """

        in_dir = self.progress_dict[band]["dir"]
        run_astro_cat = self.progress_dict[band]["run_astro_cat"]

        kws = get_kws(
            parameters=self.step_parameters,
            func=AstrometricAlignStep,
            target=self.target,
            band=band,
            max_level=0,
        )

        catalogs = kws["catalogs"]
        align_mapping = kws["align_mapping"]

        align_mapping_mode = kws["align_mapping_mode"]
        tweakreg_parameters = kws["tweakreg_parameters"]
        reproject_func = kws["reproject_func"]
        overwrite = kws["overwrite"]

        if reproject_func not in ALLOWED_REPROJECT_FUNCS:
            raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")

        step_complete_file = os.path.join(
            in_dir,
            "astrometric_align_step_complete.txt",
        )

        if overwrite:
            os.system(f"rm -rf {os.path.join(in_dir, '*_align.fits')}")
            os.system(f"rm -rf {step_complete_file}")

        # Check if we've already run the step
        if os.path.exists(step_complete_file):
            log.info("Step already run")
            return True

        # If we're matching to pre-aligned image
        if band in align_mapping:
            success = self.align_to_aligned_image(band=band,
                                                  in_dir=in_dir,
                                                  align_mapping=align_mapping,
                                                  align_mapping_mode=align_mapping_mode,
                                                  reproject_func=reproject_func,
                                                  )

        # If we're doing a more traditional tweakreg
        else:
            if run_astro_cat:
                cat_suffix = "astro_cat.fits"
            else:
                cat_suffix = "cat.ecsv"

            success = self.tweakreg_align(band=band,
                                          in_dir=in_dir,
                                          catalogs=catalogs,
                                          cat_suffix=cat_suffix,
                                          tweakreg_parameters=tweakreg_parameters,
                                          )

        # If not everything has succeeded, then return a warning
        if not success:
            log.warning("Failures detected in astrometric alignment")
            return False

        with open(step_complete_file, "w+") as f:
            f.close()

        return True

    def align_to_aligned_image(
            self,
            band,
            in_dir,
            align_mapping=None,
            align_mapping_mode="shift",
            reproject_func="interp",
    ):
        """Align to a pre-aligned image

        This will align to a pre-aligned image, either using cross-correlation
        or by pulling out the shift values and matrix from tweakreg (default)

        Args:
            band: Band to consider
            in_dir: Input directory
            align_mapping: Mapping to use to align to
            align_mapping_mode: If locking to other JWST image,
                method to use. Option is "shift" (pull the
                tweakreg solution from the existing file),
                or "cross-corr" (do some cross-correlation
                between the images)
            reproject_func: Which reproject function to use.
                Defaults to 'interp'
        """

        if reproject_func == "interp":
            r_func = reproject_interp
        elif reproject_func == "exact":
            r_func = reproject_exact
        elif reproject_func == "adaptive":
            r_func = reproject_adaptive
        else:
            raise ValueError(f"reproject_func should be one of {ALLOWED_REPROJECT_FUNCS}")

        if align_mapping is None:
            raise ValueError("Require an alignment mapping to map to")

        files = glob.glob(
            os.path.join(
                in_dir,
                f"*{self.step_ext}.fits",
            ),
        )

        if len(files) == 0:
            log.warning("No files found to align")
            return True

        log.info("Aligning to pre-aligned image")

        ref_band = align_mapping[band]
        ref_band_type = get_band_type(ref_band)

        ref_hdu_name = os.path.join(
            self.target_dir,
            ref_band,
            "lv3",
            f"{self.target.lower()}_{ref_band_type}_lv3_{ref_band.lower()}_i2d_align.fits",
        )

        if not os.path.exists(ref_hdu_name):
            log.warning(f"reference HDU {ref_hdu_name} not found. Will just rename files")

        for file in files:
            log.info(f"Aligning {os.path.split(file)[-1]}")

            aligned_file = file.replace(
                f"{self.step_ext}.fits",
                f"{self.step_ext}_align.fits",
            )

            if not os.path.exists(ref_hdu_name):
                os.system(f"cp {file} {aligned_file}")
                continue

            with datamodels.open(ref_hdu_name) as ref_im:
                # Get the WCS, either from lv3 or the HDU
                try:
                    ref_wcs = get_lv3_wcs(ref_im)
                except ValueError:
                    with fits.open(ref_hdu_name) as hdu:
                        ref_wcs = WCS(hdu["SCI"])

                ref_data = copy.deepcopy(ref_im.data)
                ref_err = copy.deepcopy(ref_im.err)
                ref_data[ref_data == 0] = np.nan

                # For shifts, pull these things out
                shift = ref_im.meta.abs_astro_alignment.shift
                matrix = ref_im.meta.abs_astro_alignment.matrix

                # Cast these to numpy, so they can be pickled properly later
                shift = shift.astype(np.ndarray).astype(float)
                matrix = matrix.astype(np.ndarray).astype(float)

                with datamodels.open(file) as target_im:
                    target_wcs = get_lv3_wcs(target_im)
                    target_wcs_corrector = FITSWCSCorrector(target_wcs)
                    target_wcs_corrector_orig = copy.deepcopy(target_wcs_corrector)
                    target_data = copy.deepcopy(target_im.data)
                    target_err = copy.deepcopy(target_im.err)
                    target_data[target_data == 0] = np.nan

                    if align_mapping_mode == "cross_corr":
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ref_data = r_func(
                                (ref_data, ref_wcs),
                                target_wcs,
                                shape_out=target_data.shape,
                                return_footprint=False,
                            )

                            ref_err = r_func(
                                (ref_err, ref_wcs),
                                target_wcs,
                                shape_out=target_data.shape,
                                return_footprint=False,
                            )

                        nan_idx = np.logical_or(np.isnan(ref_data), np.isnan(target_data))

                        ref_data[nan_idx] = np.nan
                        target_data[nan_idx] = np.nan

                        ref_err[nan_idx] = np.nan
                        target_err[nan_idx] = np.nan

                        # Make sure we're square, since apparently this causes weirdness
                        data_size_min = min(target_data.shape)
                        data_slice_i = slice(
                            target_data.shape[0] // 2 - data_size_min // 2,
                            target_data.shape[0] // 2 + data_size_min // 2,
                        )
                        data_slice_j = slice(
                            target_data.shape[1] // 2 - data_size_min // 2,
                            target_data.shape[1] // 2 + data_size_min // 2,
                        )

                        x_off, y_off = cross_correlation_shifts(
                            ref_data[data_slice_i, data_slice_j],
                            target_data[data_slice_i, data_slice_j],
                            errim1=ref_err[data_slice_i, data_slice_j],
                            errim2=target_err[data_slice_i, data_slice_j],
                        )
                        shift = [-x_off, -y_off]
                        matrix = [[1, 0], [0, 1]]

                        log.info(f"Found offset of {shift}")

                    elif align_mapping_mode == "shift":
                        # Add in shift metadata
                        target_im.meta.abs_astro_alignment = {
                            "shift": shift,
                            "matrix": matrix,
                        }

                    # Apply correction
                    target_wcs_corrector.set_correction(
                        shift=shift,
                        matrix=matrix,
                        ref_tpwcs=target_wcs_corrector_orig,
                    )

                    target_hdr, new_gwcs = transform_wcs_gwcs(target_wcs_corrector.wcs)
                    target_im.meta.wcs = new_gwcs

                    # Update WCS info
                    updated_im = lv3_update_fits_wcsinfo(im=target_im, hdr=target_hdr)

                    updated_im.write(aligned_file)

            # Also apply this to each individual crf file
            crf_files = glob.glob(
                os.path.join(
                    in_dir,
                    "*_crf.fits",
                )
            )

            crf_files.sort()

            if len(crf_files) > 0:
                successes = self.move_tweakback_files(
                    crf_files,
                    shift=shift,
                    matrix=matrix,
                    ref_tpwcs=target_wcs_corrector_orig,
                )
                if not np.all(successes):
                    log.warning("Not all crf files tweakbacked. May cause issues")

            del updated_im

        return True

    def tweakreg_align(
            self,
            band,
            in_dir,
            catalogs=None,
            cat_suffix="cat.ecsv",
            tweakreg_parameters=None,
    ):
        """Align using tweakreg

        Args:
            band: Band to consider
            in_dir: Input directory
            catalogs: Dictionary for the external alignment
                catalogs
            cat_suffix: extension for the existing
                catalog. Defaults to "cat.ecsv",
                which is the pipeline default
            tweakreg_parameters: Dictionary of parameters
                to pass to tweakreg for the standard alignment
        """

        files = glob.glob(
            os.path.join(
                in_dir,
                f"*{self.step_ext}.fits",
            ),
        )

        if len(files) == 0:
            log.warning("No files found to align")
            return True

        if catalogs is None:
            catalogs = {}
        if tweakreg_parameters is None:
            tweakreg_parameters = {}

        if self.target not in catalogs:
            log.warning("astrometric_alignment_table should be set!")
            return True

        log.info("Aligning to external catalog")

        align_catalog = os.path.join(
            self.catalog_dir,
            catalogs[self.target],
        )

        if not os.path.exists(align_catalog):
            log.warning("Requested astrometric alignment table not found!")
            return True

        align_table = QTable.read(align_catalog, format="fits")
        ref_tab = Table()

        ref_tab["RA"] = align_table["ra"]
        ref_tab["DEC"] = align_table["dec"]

        if "xcentroid" in align_table.colnames:
            ref_tab["xcentroid"] = align_table["xcentroid"]
            ref_tab["ycentroid"] = align_table["ycentroid"]

        for file in files:
            aligned_file = file.replace(".fits", "_align.fits")
            aligned_table = aligned_file.replace(".fits", "_table.fits")

            # Read in the source catalogue from the pipeline

            source_cat_name = file.replace(f"{self.step_ext}.fits", cat_suffix)

            if cat_suffix.split(".")[-1] == "ecsv":
                sources = Table.read(source_cat_name, format="ascii.ecsv")
                # convenience for CARTA viewing.
                sources.write(source_cat_name.replace(".ecsv", ".fits"), overwrite=True)
            else:
                sources = Table.read(source_cat_name)

            # Filter out extended sources
            if "is_extended" in sources.colnames:
                sources = sources[~sources["is_extended"]]

            # Load in the datamodel, and pull in WCS to correct
            target_im = datamodels.open(file)
            target_wcs = get_lv3_wcs(target_im)
            target_wcs_corrector = FITSWCSCorrector(target_wcs)

            # Make a copy since we'll be overwriting this along the way
            target_wcs_corrector_orig = copy.deepcopy(target_wcs_corrector)

            # Parse down the table and convert appropriately
            target_tab = Table()

            # Get TPx/y out -- do everything in pixel space
            target_tab["TPx"], target_tab["TPy"] = target_wcs_corrector.world_to_det(
                sources["sky_centroid"].ra,
                sources["sky_centroid"].dec,
            )
            ref_tab["TPx"], ref_tab["TPy"] = target_wcs_corrector.world_to_det(
                ref_tab["RA"],
                ref_tab["DEC"],
            )

            # We'll also need x and y for later
            target_tab["x"] = sources["xcentroid"]
            target_tab["y"] = sources["ycentroid"]

            target_tab["ra"] = sources["sky_centroid"].ra.value
            target_tab["dec"] = sources["sky_centroid"].dec.value

            # Do the fit -- potentially take an iterative approach, using
            # multiple homing-in iterations
            multiple_iterations = False
            n_iterations = 0
            for key in tweakreg_parameters.keys():
                if "iteration" in key:
                    multiple_iterations = True
                    n_iterations += 1

            if not multiple_iterations:
                n_iterations = 1

            wcs_aligned_fit = None

            xoffset, yoffset = 0, 0
            shift = np.array([0, 0])
            matrix = np.array([[1, 0], [0, 1]])

            for iteration in range(n_iterations):
                # Make sure we're not overwriting WCS
                target_wcs_corrector = copy.deepcopy(target_wcs_corrector_orig)

                if not multiple_iterations:
                    astrometry_parameter_dict = copy.deepcopy(tweakreg_parameters)
                else:
                    astrometry_parameter_dict = copy.deepcopy(
                        tweakreg_parameters[f"iteration{iteration + 1:d}"]
                    )

                # Run a match
                match = XYXYMatch(
                    xoffset=xoffset,
                    yoffset=yoffset,
                )
                for key in astrometry_parameter_dict.keys():
                    value = parse_parameter_dict(
                        astrometry_parameter_dict,
                        key,
                        band,
                        self.target,
                    )
                    if value == "VAL_NOT_FOUND":
                        continue

                    recursive_setattr(match, key, value, protected=True)

                ref_idx, target_idx = match(
                    ref_tab,
                    target_tab,
                    tp_units="pix",
                )

                fit_wcs_args = get_default_args(fit_wcs)

                fit_wcs_kws = {}
                for fit_wcs_arg in fit_wcs_args.keys():
                    if fit_wcs_arg in astrometry_parameter_dict.keys():
                        arg_val = parse_parameter_dict(
                            astrometry_parameter_dict,
                            fit_wcs_arg,
                            band,
                            self.target,
                        )
                        if arg_val == "VAL_NOT_FOUND":
                            arg_val = fit_wcs_args[fit_wcs_arg]
                    else:
                        arg_val = fit_wcs_args[fit_wcs_arg]

                    # sigma here is fiddly, test if it's a tuple and fix to rmse if not
                    if fit_wcs_arg == "sigma":
                        if type(arg_val) != tuple:
                            arg_val = (arg_val, "rmse")

                    fit_wcs_kws[fit_wcs_arg] = arg_val

                # Do alignment
                try:
                    wcs_aligned_fit = fit_wcs(
                        refcat=ref_tab[ref_idx],
                        imcat=target_tab[target_idx],
                        corrector=target_wcs_corrector,
                        **fit_wcs_kws,
                    )

                    # Pull out offsets, remember there's a negative here to the shift
                    xoffset, yoffset = -wcs_aligned_fit.meta["fit_info"]["shift"]

                    # Pull out shifts and matrix
                    shift = wcs_aligned_fit.meta["fit_info"]["shift"]
                    matrix = wcs_aligned_fit.meta["fit_info"]["matrix"]

                except ValueError:
                    log.warning("No catalog matches found. Defaulting to no shift")

                    # Reset everything to avoid crashes
                    wcs_aligned_fit = None
                    xoffset, yoffset = 0, 0
                    shift = np.array([0, 0])
                    matrix = np.array([[1, 0], [0, 1]])

            target_wcs_corrected = copy.deepcopy(target_wcs_corrector_orig)

            # Put the correction in and properly update header.
            target_wcs_corrected.set_correction(
                shift=shift,
                matrix=matrix,
                ref_tpwcs=target_wcs_corrector_orig,
            )

            target_hdr, new_gwcs = transform_wcs_gwcs(target_wcs_corrected.wcs)
            target_im.meta.wcs = new_gwcs

            # Add in shift metadata
            target_im.meta.abs_astro_alignment = {
                "shift": shift,
                "matrix": matrix,
            }

            # Update WCS info
            target_im = lv3_update_fits_wcsinfo(
                im=target_im,
                hdr=target_hdr,
            )

            target_im.write(aligned_file)

            # Also apply this to each individual crf file
            crf_files = glob.glob(
                os.path.join(
                    in_dir,
                    "*_crf.fits",
                )
            )

            crf_files.sort()

            if len(crf_files) > 0:
                successes = self.move_tweakback_files(
                    crf_files,
                    shift=shift,
                    matrix=matrix,
                    ref_tpwcs=target_wcs_corrector_orig,
                )
                if not np.all(successes):
                    log.warning("Not all crf files tweakbacked. May cause issues")

            if wcs_aligned_fit is not None:
                fit_info = wcs_aligned_fit.meta["fit_info"]
                fit_mask = fit_info["fitmask"]

                # Pull out useful alignment info to the table -- HST x/y/RA/Dec, JWST x/y/RA/Dec (corrected and
                # uncorrected)
                aligned_tab = Table()

                # Catch if there's only RA/Dec in the reference table
                if "xcentroid" in ref_tab.colnames:
                    aligned_tab["xcentroid_ref"] = ref_tab[ref_idx]["xcentroid"][
                        fit_mask
                    ]
                    aligned_tab["ycentroid_ref"] = ref_tab[ref_idx]["ycentroid"][
                        fit_mask
                    ]
                aligned_tab["ra_ref"] = ref_tab[ref_idx]["RA"][fit_mask]
                aligned_tab["dec_ref"] = ref_tab[ref_idx]["DEC"][fit_mask]

                # Since we're pulling from the source catalogue, these should all exist
                aligned_tab["xcentroid_jwst"] = target_tab[target_idx]["x"][fit_mask]
                aligned_tab["ycentroid_jwst"] = target_tab[target_idx]["y"][fit_mask]
                aligned_tab["ra_jwst_uncorr"] = target_tab[target_idx]["ra"][fit_mask]
                aligned_tab["dec_jwst_uncorr"] = target_tab[target_idx]["dec"][fit_mask]

                aligned_tab["ra_jwst_corr"] = fit_info["fit_RA"]
                aligned_tab["dec_jwst_corr"] = fit_info["fit_DEC"]

                aligned_tab.write(aligned_table, format="fits", overwrite=True)

            else:
                log.warning("Fit unsuccessful, not writing out table")

        return True

    def move_tweakback_files(
            self,
            files,
            shift=None,
            matrix=None,
            ref_tpwcs=None,
    ):
        """Wrapper to parallelise up tweakback

        Args:
            files: List of files to tweakback
            shift: shift for tweakback. Defaults
                to None
            matrix: rotation/skew matrix. Defaults
                to None
            ref_tpwcs: WCS defining the plane in which
                the shift/matrix was defined. Defaults
                to None
        """

        log.info("Running tweakback")

        procs = np.nanmin([self.procs, len(files)])

        with mp.get_context("fork").Pool(procs) as pool:
            successes = []

            for success in tqdm(
                    pool.imap_unordered(
                        partial(
                            parallel_tweakback,
                            shift=shift,
                            matrix=matrix,
                            ref_tpwcs=ref_tpwcs,
                        ),
                        files,
                    ),
                    total=len(files),
                    ascii=True,
                    desc="tweakback",
            ):
                successes.append(success)

            pool.close()
            pool.join()
            gc.collect()

        return successes
