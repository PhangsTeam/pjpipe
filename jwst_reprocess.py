import copy
import functools
import glob
import inspect
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
import warnings
from functools import partial
from multiprocessing import cpu_count

import numpy as np
from astropy.io import fits
from astropy.nddata.bitmask import interpret_bit_flags, bitfield_to_boolean_mask
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, QTable
from drizzlepac import updatehdr
from image_registration import cross_correlation_shifts
from photutils.segmentation import make_source_mask
from photutils.detection import DAOStarFinder
from reproject import reproject_interp
from stwcs.wcsutil import HSTWCS
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from tweakwcs import fit_wcs, XYXYMatch
from tweakwcs.correctors import FITSWCSCorrector, JWSTWCSCorrector

from nircam_destriping import NircamDestriper

jwst = None
datamodels = None
pixel = None
update_fits_wcsinfo = None
calwebb_detector1 = None
calwebb_image2 = None
calwebb_image3 = None
TweakRegStep = None
SkyMatchStep = None
SourceCatalogStep = None

# Pipeline steps
ALLOWED_STEPS = [
    'lv1',
    'lv2',
    'destripe',
    'bg_sub',
    'dither_match',
    'lyot_adjust',
    'wcs_adjust',
    'lv3',
    'astrometric_align',
]

# Pipeline steps where we don't want to delete the whole directory
STEP_NO_DEL_DIR = [
    'astrometric_align',
]

# All NIRCAM bands
NIRCAM_BANDS = [
    'F070W',
    'F090W',
    'F115W',
    'F140M',
    'F150W',
    'F162M',
    'F164N',
    'F150W2',
    'F182M',
    'F187N',
    'F200W',
    'F210M',
    'F212N',
    'F250M',
    'F277W',
    'F300M',
    'F322W2',
    'F323N',
    'F335M',
    'F356W',
    'F360M',
    'F405N',
    'F410M',
    'F430M',
    'F444W',
    'F460M',
    'F466N',
    'F470N',
    'F480M',
]

# All MIRI bands
MIRI_BANDS = [
    'F560W',
    'F770W',
    'F1000W',
    'F1130W',
    'F1280W',
    'F1500W',
    'F1800W',
    'F2100W',
    'F2550W',
]

BAND_EXTS = {'nircam': 'nrc*',
             'miri': 'mirimage'}

# FWHM of bands in pixels
FWHM_PIX = {
    # NIRCAM
    'F070W': 0.987,
    'F090W': 1.103,
    'F115W': 1.298,
    'F140M': 1.553,
    'F150W': 1.628,
    'F162M': 1.770,
    'F164N': 1.801,
    'F150W2': 1.494,
    'F182M': 1.990,
    'F187N': 2.060,
    'F200W': 2.141,
    'F210M': 2.304,
    'F212N': 2.341,
    'F250M': 1.340,
    'F277W': 1.444,
    'F300M': 1.585,
    'F322W2': 1.547,
    'F323N': 1.711,
    'F335M': 1.760,
    'F356W': 1.830,
    'F360M': 1.901,
    'F405N': 2.165,
    'F410M': 2.179,
    'F430M': 2.300,
    'F444W': 2.302,
    'F460M': 2.459,
    'F466N': 2.507,
    'F470N': 2.535,
    'F480M': 2.574,
    # MIRI
    'F560W': 1.636,
    'F770W': 2.187,
    'F1000W': 2.888,
    'F1130W': 3.318,
    'F1280W': 3.713,
    'F1500W': 4.354,
    'F1800W': 5.224,
    'F2100W': 5.989,
    'F2550W': 7.312,
}


def parallel_destripe(hdu_name,
                      band,
                      target,
                      destripe_parameter_dict=None,
                      pca_dir=None,
                      out_dir=None,
                      plot_dir=None,
                      ):
    """Function to parallelise destriping"""

    if destripe_parameter_dict is None:
        destripe_parameter_dict = {}

    out_file = os.path.join(out_dir, os.path.split(hdu_name)[-1])

    if os.path.exists(out_file):
        return True

    if pca_dir is not None:
        pca_file = os.path.join(pca_dir,
                                os.path.split(hdu_name)[-1].replace('.fits', '_pca.pkl')
                                )
    else:
        pca_file = None

    nc_destripe = NircamDestriper(hdu_name=hdu_name,
                                  hdu_out_name=out_file,
                                  pca_file=pca_file,
                                  plot_dir=plot_dir,
                                  )

    for key in destripe_parameter_dict.keys():

        value = parse_parameter_dict(destripe_parameter_dict,
                                     key,
                                     band,
                                     target,
                                     )
        if value == 'VAL_NOT_FOUND':
            continue

        recursive_setattr(nc_destripe, key, value)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with threadpool_limits(limits=1, user_api='blas'):
            nc_destripe.run_destriping()

    return True


def parallel_tweakback(crf_file,
                       matrix=None,
                       shift=None,
                       ref_tpwcs=None,
                       ):
    """Wrapper function to parallelise tweakback routine"""

    if matrix is None:
        matrix = [[1, 0], [0, 1]]
    if shift is None:
        shift = [0, 0]

    crf_input_im = datamodels.open(crf_file)
    model_name = os.path.splitext(crf_input_im.meta.filename)[0].strip('_- ')
    refang = crf_input_im.meta.wcsinfo.instance

    crf_im = JWSTWCSCorrector(
        wcs=crf_input_im.meta.wcs,
        wcsinfo={
            'roll_ref': refang['roll_ref'],
            'v2_ref': refang['v2_ref'],
            'v3_ref': refang['v3_ref']
        },
        meta={
            'image_model': crf_input_im,
            'name': model_name
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        crf_im.set_correction(matrix=matrix,
                              shift=shift,
                              ref_tpwcs=ref_tpwcs,
                              )

        crf_input_im = crf_im.meta['image_model']
        crf_input_im.meta.wcs = crf_im.wcs

        try:
            update_fits_wcsinfo(
                crf_input_im,
                max_pix_error=0.01,
                npoints=16,
            )
        except (ValueError, RuntimeError) as e:
            logging.warning(
                "Failed to update 'meta.wcsinfo' with FITS SIP "
                f'approximation. Reported error is:\n"{e.args[0]}"'
            )
            return False

    crf_out_file = crf_file.replace('.fits', '_tweakback.fits')
    crf_input_im.save(crf_out_file)

    return True


def sigma_clip(data,
               dq_mask=None,
               sigma=1.5,
               n_pixels=5,
               max_iterations=20,
               ):
    """Get sigma-clipped statistics for data"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = make_source_mask(data, mask=dq_mask, nsigma=sigma, npixels=n_pixels)
        if dq_mask is not None:
            mask = np.logical_or(mask, dq_mask)
        mean, median, std_dev = sigma_clipped_stats(data, mask=mask, sigma=sigma, maxiters=max_iterations)

    return mean, median, std_dev


def background_subtract(hdu_filename,
                        sigma=1.5,
                        n_pixels=5,
                        max_iterations=20,
                        ):
    """Sigma-clipped background subtraction for fits HDU"""

    hdu = fits.open(hdu_filename)

    dq_mask = hdu['DQ'].data != 0

    mean, median, std = sigma_clip(hdu['SCI'].data,
                                   dq_mask=dq_mask,
                                   sigma=sigma,
                                   n_pixels=n_pixels,
                                   max_iterations=max_iterations,
                                   )
    hdu['SCI'].data -= median

    return hdu


def background_match(files,
                     out_dir,
                     remove_background=False,
                     ):
    """Match a series of observations to a common level

    Args:
        files: List of files to match
        out_dir: Directory to save the subtracted files to
        remove_background (bool): Whether to remove a calculated background, or just to match to the first tile in
            the sequence. Defaults to False
    """

    hdus = []

    for in_file in files:
        hdu = datamodels.open(in_file)
        hdus.append(hdu)

    dq_bits = interpret_bit_flags('~DO_NOT_USE+NON_SCIENCE', flag_name_map=pixel)

    # Reproject the tiles and DQ arrays to the first pixel grid
    data_reproj = np.full([*hdus[0].data.shape, len(files)], np.nan)

    data_reproj[:, :, 0] = copy.deepcopy(hdus[0].data)

    sci_mask = np.zeros_like(hdus[0].data)
    dq_mask = np.zeros_like(hdus[0].data)

    sci_mask[np.logical_or(hdus[0].data == 0,
                           ~np.isfinite(hdus[0].data)
                           )] = 1
    dq_bit_mask = bitfield_to_boolean_mask(
        hdus[0].dq.astype(np.uint8),
        dq_bits,
        good_mask_value=0,
        dtype=np.uint8
    )
    dq_mask[dq_bit_mask != 0] = 1

    # Calculate a "background value", to potentially subtract later
    bg_val = sigma_clip(data_reproj[:, :, 0],
                        dq_mask=dq_mask,
                        )[1]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        wcs = hdus[0].meta.wcs.to_fits_sip()
        for i, hdu in enumerate(hdus[1:]):

            hdu_wcs = hdu.meta.wcs.to_fits_sip()
            data_reproj[:, :, i + 1] = reproject_interp((hdu.data, hdu_wcs),
                                                        wcs,
                                                        return_footprint=False,
                                                        )

            # Do this for DQ since the WCS are different
            dq = fits.ImageHDU(hdu.dq, wcs)
            dq_reproj = reproject_interp(dq,
                                         wcs,
                                         order='nearest-neighbor',
                                         return_footprint=False,
                                         )
            dq_reproj[~np.isfinite(dq_reproj)] = 0

            # Catch any 0s/NaNs in the science image
            sci_mask[np.logical_or(data_reproj[:, :, i + 1] == 0,
                                   ~np.isfinite(data_reproj[:, :, i + 1])
                                   )] = 1

            # Convert the DQ non-science/do not use to another mask
            dq_bit_mask = bitfield_to_boolean_mask(
                dq_reproj.astype(np.uint8),
                dq_bits,
                good_mask_value=0,
                dtype=np.uint8
            )
            dq_mask[dq_bit_mask != 0] = 1

    # Get rid of any pixels masked in any DQ mask or missing in any science image
    for i in range(data_reproj.shape[-1]):
        data_reproj[:, :, i][dq_mask != 0] = np.nan
        data_reproj[:, :, i][sci_mask != 0] = np.nan

    # Minimize the differences between each tile and the first
    for i in range(len(hdus)):

        diff = data_reproj[:, :, i] - data_reproj[:, :, 0]
        diff = diff.flatten()

        # Sigma-clip to remove outliers in the distribution
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            diff_val = sigma_clipped_stats(diff, maxiters=None)[1]

        hdus[i].data -= diff_val
        # if hdus[i].meta.background.level is not None:
        #     hdus[i].meta.background.level += diff_val
        # else:
        #     hdus[i].meta.background.level = diff_val

        # Remove the pre-calculated background if we want that
        if remove_background:
            hdus[i].data -= bg_val

        #     if hdus[i].meta.background.level is not None:
        #         hdus[i].meta.background.level += bg_val
        #     else:
        #         hdus[i].meta.background.level = bg_val
        #
        # hdus[i].meta.background.subtracted = True

        hdu_out_name = os.path.join(out_dir, os.path.split(files[i])[-1])
        hdus[i].save(hdu_out_name, overwrite=True)

    return True


def parse_fits_to_table(file,
                        check_bgr=False,
                        check_type='parallel_off',
                        background_name='off',
                        ):
    """Pull necessary info out of fits headers

    Args:
        file (str): File to get info for
        check_bgr (bool): Whether to check if this is a science or background observation (in the MIRI case)
        check_type (str): How to check if background observation. Options are 'parallel_off', which will use the
            filename to see if it's a parallel observation with NIRCAM, or 'check_in_name', which will use the
            observation name to check, matching against 'background_name'. Defaults to 'parallel_off'
        background_name (str): Name to indicate background observation. Defaults to 'off'.
    """

    if check_bgr:

        f_type = 'sci'

        if check_type == 'parallel_off':

            file_split = os.path.split(file)[-1]

            if file_split.split('_')[1][2] == '2':
                f_type = 'bgr'

        elif check_type == 'check_in_name':
            hdu = fits.open(file)
            if background_name in hdu[0].header['TARGPROP'].lower():
                f_type = 'bgr'
            hdu.close()
        else:
            raise Warning('check_type %s not known' % check_type)
    else:
        f_type = 'sci'
    with fits.open(file) as hdu:
        return file, f_type, hdu[0].header['OBSERVTN'], hdu[0].header['filter'], \
            hdu[0].header['DATE-BEG'], hdu[0].header['DURATION'], \
            hdu[0].header['OBSLABEL'].lower().strip(), hdu[0].header['PROGRAM']


def parse_parameter_dict(parameter_dict,
                         key,
                         band,
                         target,
                         ):
    """Pull values out of a parameter dictionary

    Args:
        parameter_dict (dict): Dictionary of parameters and associated values
        key (str): Particular key in parameter_dict to consider
        band (str): JWST band, to parse out band type and potentially per-band
            values
        target (str): JWST target, for very specific values

    """

    value = parameter_dict[key]

    if band in MIRI_BANDS:
        band_type = 'miri'
        short_long = 'miri'
        pixel_scale = 0.11
    elif band in NIRCAM_BANDS:
        band_type = 'nircam'

        # Also pull out the distinction between short and long NIRCAM
        if int(band[1:4]) <= 212:
            short_long = 'nircam_short'
            pixel_scale = 0.031
        else:
            short_long = 'nircam_long'
            pixel_scale = 0.063
    else:
        raise Warning('Band type %s not known!' % band)

    if isinstance(value, dict):

        # Define a priority here. It goes:
        # * target
        # * band
        # * nircam_short/nircam_long
        # * nircam/miri

        if target in value.keys():
            value = value[target]

        elif band in value.keys():
            value = value[band]

        elif band_type == 'nircam' and short_long in value.keys():
            value = value[short_long]

        elif band_type in value.keys():
            value = value[band_type]

        else:
            value = 'VAL_NOT_FOUND'

        # Add another level, where we can specify this per-target
        if isinstance(value, dict):
            if target in value.keys():
                value = value[target]
            else:
                value = 'VAL_NOT_FOUND'

    # Finally, if we have a string with a 'pix' in there, we need to convert to arcsec
    if isinstance(value, str):
        if 'pix' in value:
            value = float(value.strip('pix')) * pixel_scale

    return value


def recursive_setattr(f,
                      attribute,
                      value,
                      protected=False,
                      ):
    pre, _, post = attribute.rpartition('.')

    if pre:
        pre_exists = True
    else:
        pre_exists = False

    if protected:
        post = '_' + post
    return setattr(recursive_getattr(f, pre) if pre_exists else f, post, value)


def recursive_getattr(f,
                      attribute,
                      *args,
                      ):
    def _getattr(f, attribute):
        return getattr(f, attribute, *args)

    return functools.reduce(_getattr, [f] + attribute.split('.'))


def attribute_setter(pipeobj,
                     parameter_dict,
                     band,
                     target,
                     ):
    for key in parameter_dict.keys():
        if type(parameter_dict[key]) is dict:
            for subkey in parameter_dict[key]:
                value = parse_parameter_dict(parameter_dict[key],
                                             subkey,
                                             band,
                                             target,
                                             )
                if value == 'VAL_NOT_FOUND':
                    continue

                recursive_setattr(pipeobj,
                                  '.'.join([key, subkey]),
                                  value,
                                  )

        else:

            value = parse_parameter_dict(parameter_dict,
                                         key,
                                         band,
                                         target,
                                         )
            if value == 'VAL_NOT_FOUND':
                continue

            recursive_setattr(pipeobj,
                              key,
                              value,
                              )
    return pipeobj


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class JWSTReprocess:

    def __init__(self,
                 target,
                 raw_dir,
                 reprocess_dir,
                 crds_dir,
                 bands=None,
                 steps=None,
                 overwrites=None,
                 obs_to_skip=None,
                 extra_obs_to_include=None,
                 lv1_parameter_dict='phangs',
                 lv2_parameter_dict='phangs',
                 lv3_parameter_dict='phangs',
                 bg_sub_parameter_dict=None,
                 destripe_parameter_dict='phangs',
                 astrometry_parameter_dict='phangs',
                 lyot_method='mask',
                 tweakreg_create_custom_catalogs=None,
                 group_tweakreg_dithers=None,
                 group_skymatch_dithers=None,
                 degroup_skymatch_dithers=None,
                 bgr_check_type='parallel_off',
                 bgr_background_name='off',
                 bgr_observation_types=None,
                 astrometric_alignment_type='image',
                 astrometric_alignment_image=None,
                 astrometric_alignment_table=None,
                 alignment_mapping=None,
                 wcs_adjust_dict=None,
                 correct_lv1_wcs=False,
                 crds_url='https://jwst-crds.stsci.edu',
                 procs=None,
                 updated_flats_dir=None,
                 process_bgr_like_science=False,
                 use_field_in_lev3=None
                 ):
        """JWST reprocessing routines.

        Will run through whole JWST pipeline, allowing for fine-tuning along the way.

        It's worth talking a little about how parameter dictionaries are passed. They should be of the form

                {'parameter': value}

        where parameter is how the pipeline names it, e.g. 'save_results', 'tweakreg.fitgeometry'. Because you might
        want to break these out per observing mode, you can also pass a dict, like

                {'parameter': {'miri': miri_val, 'nircam': nircam_val}}

        where the acceptable variants are 'miri', 'nircam', 'nircam_long', 'nircam_short', and a specific filter. As
        many bits of the pipeline require a number in arcsec rather than pixels, you can pass a value as 'Xpix', and it
        will parse according to the band you're processing.

        Args:
            * target (str): Target to run reprocessing for
            * raw_dir (str): Path to raw data
            * reprocess_dir (str): Path to reprocess data into
            * crds_dir (str): Path to CRDS data
            * bands (list): JWST filters to loop over
            * steps (list or dict): Steps to perform in the order they should be performed. Should be drawn from
                ALLOWED_STEPS. Can specify separately for NIRCam and MIRI. Defaults to None, which will run the standard
                STScI pipeline
            * overwrites (list or dict): Steps to overwrite. Should be drawn from ALLOWED_STEPS. Can be specified
                separately for NIRCam and MIRI. Defaults to None, which will not overwrite anything
            * obs_to_skip (list): Flag failed observations that may still be included in archive downloads. If the
                filename matches part of any of the strings provided here, will not include that observation in
                reprocessing. Defaults to None
            * extra_obs_to_include (dict): Dictionary in the form {target: {other_target: [obs_to_match]}}. Will then
                include observations from other targets in the reprocessing. Generally you shouldn't need to use this,
                but may be useful for taking background observations from another target. Defaults to None
            * lv1_parameter_dict (dict): Dictionary of parameters to feed to level 1 pipeline. See description above
                for how this should be formatted. Defaults to 'phangs', which will use the parameters for the
                PHANGS-JWST reduction. To keep pipeline default, use 'None'
            * lv2_parameter_dict (dict): As `lv1_parameter_dict`, but for the level 2 pipeline
            * lv3_parameter_dict (dict): As `lv1_parameter_dict`, but for the level 3 pipeline
            * bg_sub_parameter_dict (dict): As `lv1_parameter_dict`, but for the background subtraction procedure
            * destripe_parameter_dict (dict): As `lv1_parameter_dict`, but for the destriping procedure
            * astrometry_parameter_dict (dict): As `lv1_parameter_dict`, but for astrometric alignment
            * lyot_method (str): Method to account for mistmatch lyot coronagraph in MIRI imaging. Can either mask with
                `mask`, or adjust to main chip with `adjust`. Defaults to `mask`
            * tweakreg_create_custom_catalogs (list): Whether to use the SourceCatalogStep for tweakreg source finding,
                rather than the default algorithm. Should be list of 'nircam_short', 'nircam_long', 'miri'. Defaults
                to None, which will not tie any together
            * group_tweakreg_dithers (list): Which type of observations to group in Tweakreg. Should be list of
                'nircam_short', 'nircam_long', 'miri'. Defaults to None, which will not tie any together
            * group_skymatch_dithers (list): Which type of observations to group in Skymatch. Should be list of
                'nircam_short', 'nircam_long', 'miri'. Defaults to None, which will not tie any together
            * degroup_skymatch_dithers (list): Which type of observations to degroup in Skymatch. Should be list of
                'nircam_short', 'nircam_long', 'miri'. Will only do anything for short NIRCam observations. Defaults to
                None, which will not tie any together
            * bgr_check_type (str): Method to check if MIRI obs is science or background. Options are 'parallel_off' and
                'check_in_name'. Defaults to 'parallel_off'
            * bgr_background_name (str): If `bgr_check_type` is 'check_in_name', this is the string to match
            * bgr_observation_types (list): List of observation types with dedicated backgrounds. Defaults to None, i.e.
                no observations have backgrounds
            * astrometric_alignment_type (str): Whether to align to image or table. Defaults to `image`
            * astrometric_alignment_image (str): Path to image to align astrometry to
            * astrometric_alignment_table (str): Path to table to align astrometry to
            * alignment_mapping (dict): Dictionary to map basing alignments off cross-correlation with other aligned
                band. Should be of the form {'band': 'reference_band'}
            * wcs_adjust_dict (dict): dict to adjust image group WCS before tweakreg step. Should be of form
                {target: {'visit': {'matrix': [[1, 0], [0, 1]], 'shift': [dx, dy]}}}. Defaults to None.
            * correct_lv1_wcs (bool): Check WCS in uncal files, since there is a bug that some don't have this populated
                when pulled from the archive. Defaults to False
            * crds_url (str): URL to get CRDS files from. Defaults to 'https://jwst-crds.stsci.edu', which will be the
                latest versions of the files
            * procs (int): Number of parallel processes to run during destriping. Will default to half the number of
                cores in the system
            * updated_flats_dir (str): Directory with the updated flats to use instead of default ones.
            * process_bgr_like_science (bool): if True, than additionally process the offset images
                in the same way as the science (for testing purposes only)
            * use_field_in_lev3 (list): if not None, then should be a list of indexes corresponding to
                the number of pointings to use (jwxxxxxyyyzzz_...), where zzz is the considered numbers

        TODO:
            * Record alignment parameters into the fits header

        """

        os.environ['CRDS_SERVER_URL'] = crds_url
        os.environ['CRDS_PATH'] = crds_dir

        # Use global variables, so we can import JWST stuff preserving environment variables
        global jwst
        global calwebb_detector1, calwebb_image2, calwebb_image3
        global TweakRegStep, SkyMatchStep, SourceCatalogStep
        global datamodels, pixel, update_fits_wcsinfo

        import jwst
        from jwst import datamodels
        from jwst.assign_wcs.util import update_fits_wcsinfo
        from jwst.datamodels.dqflags import pixel
        from jwst.pipeline import calwebb_detector1, calwebb_image2, calwebb_image3
        from jwst.tweakreg import TweakRegStep
        from jwst.skymatch import SkyMatchStep
        from jwst.source_catalog import SourceCatalogStep

        self.target = target

        if bands is None:
            # Loop over all bands
            bands = NIRCAM_BANDS + MIRI_BANDS

        self.bands = bands

        self.raw_dir = raw_dir
        self.reprocess_dir = reprocess_dir

        if alignment_mapping is None:
            alignment_mapping = {}
        self.alignment_mapping = alignment_mapping

        if wcs_adjust_dict is None:
            wcs_adjust_dict = {}
        self.wcs_adjust_dict = wcs_adjust_dict

        if lv1_parameter_dict is None:
            lv1_parameter_dict = {}
        elif lv1_parameter_dict == 'phangs':
            lv1_parameter_dict = {
                'save_results': True,

                'ramp_fit.suppress_one_group': False,

                'refpix.use_side_ref_pixels': True,
            }

        self.lv1_parameter_dict = lv1_parameter_dict

        if lv2_parameter_dict is None:
            lv2_parameter_dict = {}
        elif lv2_parameter_dict == 'phangs':
            lv2_parameter_dict = {
                'save_results': True,

                'bkg_subtract.save_combined_background': True,
                'bkg_subtract.sigma': 1.5,
            }

        self.lv2_parameter_dict = lv2_parameter_dict

        if lv3_parameter_dict is None:
            lv3_parameter_dict = {}
        elif lv3_parameter_dict == 'phangs':
            lv3_parameter_dict = {
                'save_results': True,

                'tweakreg.align_to_gaia': False,
                'tweakreg.brightest': 500,
                'tweakreg.snr_threshold': 3,
                'tweakreg.expand_refcat': True,
                'tweakreg.fitgeometry': 'shift',
                'tweakreg.minobj': 3,
                'tweakreg.peakmax': {'nircam': 20, 'miri': None},
                'tweakreg.searchrad': '10pix',
                'tweakreg.separation': '10pix',
                'tweakreg.tolerance': {'nircam_short': '5pix', 'nircam_long': '10pix', 'miri': '10pix'},
                'tweakreg.use2dhist': False,

                'tweakreg.roundlo': -0.5,
                'tweakreg.roundhi': 0.5,
                # Tweak boxsize, so we detect objects in diffuse emission
                'tweakreg.bkg_boxsize': 100,

                'skymatch.skip': {'miri': False},
                'skymatch.skymethod': {'nircam': 'global+match', 'miri': 'match'},
                'skymatch.subtract': {'nircam': True, 'miri': True},
                'skymatch.skystat': 'median',
                'skymatch.match_down': {'miri': True},
                'skymatch.nclip': {'nircam': 20, 'miri': 10},
                'skymatch.lsigma': {'nircam': 3, 'miri': 1.5},
                'skymatch.usigma': {'nircam': 3, 'miri': 1.5},

                'outlier_detection.in_memory': True,

                'resample.rotation': 0.0,
                'resample.in_memory': True,

                'source_catalog.snr_threshold': 3,
                'source_catalog.npixels': 5,
                'source_catalog.bkg_boxsize': 100,
                'source_catalog.deblend': True
            }

        self.lv3_parameter_dict = lv3_parameter_dict

        if bg_sub_parameter_dict is None:
            bg_sub_parameter_dict = {}
        self.bg_sub_parameter_dict = bg_sub_parameter_dict

        if destripe_parameter_dict is None:
            destripe_parameter_dict = {}
        elif destripe_parameter_dict == 'phangs':
            destripe_parameter_dict = {
                'quadrants': True,
                'filter_diffuse': True,
                'destriping_method': 'pca',
                'dilate_size': 7,
                'pca_reconstruct_components': 5,
                'final_large_scale_subtraction': False,
            }

            # Old version, using median filter
            # destripe_parameter_dict = {
            #     'quadrants': False,
            #     'destriping_method': 'median_filter',
            #     'dilate_size': 7,
            #     'median_filter_scales': [7, 31, 63, 127, 511]
            # }

        self.destripe_parameter_dict = destripe_parameter_dict

        if astrometry_parameter_dict is None:
            astrometry_parameter_dict = {}
        elif astrometry_parameter_dict == 'phangs':
            astrometry_parameter_dict = {
                'searchrad': 10,
                'separation': 0.000001,
                'tolerance': 0.7,
                'use2dhist': True,
                'fitgeom': 'shift',
                'nclip': 3,
                'sigma': 3,
            }

        self.astrometry_parameter_dict = astrometry_parameter_dict

        self.lyot_method = lyot_method

        if tweakreg_create_custom_catalogs is None:
            tweakreg_create_custom_catalogs = []
        self.tweakreg_create_custom_catalogs = tweakreg_create_custom_catalogs

        if group_tweakreg_dithers is None:
            group_tweakreg_dithers = []
        self.group_tweakreg_dithers = group_tweakreg_dithers

        if group_skymatch_dithers is None:
            group_skymatch_dithers = []
        self.group_skymatch_dithers = group_skymatch_dithers

        if degroup_skymatch_dithers is None:
            degroup_skymatch_dithers = []
        self.degroup_skymatch_dithers = degroup_skymatch_dithers

        # Default to standard STScI pipeline
        if steps is None:
            steps = [
                'lv1',
                'lv2',
                'lv3',
            ]
        if overwrites is None:
            overwrites = []
        if obs_to_skip is None:
            obs_to_skip = []
        if extra_obs_to_include is None:
            extra_obs_to_include = {}

        self.steps = steps
        self.overwrites = overwrites
        self.obs_to_skip = obs_to_skip
        self.extra_obs_to_include = extra_obs_to_include

        self.astrometric_alignment_type = astrometric_alignment_type
        self.astrometric_alignment_image = astrometric_alignment_image
        self.astrometric_alignment_table = astrometric_alignment_table

        self.bgr_check_type = bgr_check_type
        self.bgr_background_name = bgr_background_name
        if bgr_observation_types is None:
            bgr_observation_types = []
        self.bgr_observation_types = bgr_observation_types

        self.correct_lv1_wcs = correct_lv1_wcs

        if procs is None:
            procs = cpu_count() // 2

        self.procs = procs

        if updated_flats_dir is not None and os.path.isdir(updated_flats_dir):
            self.updated_flats_dir = updated_flats_dir
        else:
            self.updated_flats_dir = None
        self.process_bgr_like_science = process_bgr_like_science
        self.use_field_in_lev3 = use_field_in_lev3
        logging.basicConfig(level=logging.INFO, format='%{name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def run_all(self):
        """Run the whole pipeline reprocess"""

        self.logger.info('Reprocessing %s' % self.target)

        in_band_dir_dict = {
            'lv1': 'uncal',
            'lv2': 'rate',
            'destripe': 'cal',
            'bg_sub': 'cal',
            'dither_match': 'cal',
            'lyot_adjust': 'cal',
            'wcs_adjust': 'cal',
            'lv3': 'cal',
            'astrometric_align': 'lv3',
        }

        out_band_dir_dict = {
            'lv1': 'rate',
            'lv2': 'cal',
            'destripe': 'destripe',
            'bg_sub': 'bg_sub',
            'dither_match': 'dither_match',
            'lyot_adjust': 'lyot_adjust',
            'wcs_adjust': 'wcs_adjust',
            'lv3': 'lv3',
            'astrometric_align': 'lv3',
        }

        step_ext_dict = {
            'lv1': 'uncal',
            'lv2': 'rate',
            'destripe': 'cal',
            'bg_sub': 'cal',
            'dither_match': 'cal',
            'lyot_adjust': 'cal',
            'wcs_adjust': 'cal',
            'lv3': 'i2d',
            'astrometric_align': 'i2d_align',
        }

        for band in self.bands:

            base_band_dir = os.path.join(self.reprocess_dir,
                                         self.target,
                                         band)
            raw_data_moved = False

            pupil = 'CLEAR'
            jwst_filter = copy.deepcopy(band)

            if band in NIRCAM_BANDS:
                band_type = 'nircam'

                # For some NIRCAM filters, we need to distinguish filter/pupil.
                # TODO: These may not be unique, so may need editing
                if band in ['F162M', 'F164N']:
                    pupil = copy.deepcopy(band)
                    jwst_filter = 'F150W2'
                if band == 'F323N':
                    pupil = copy.deepcopy(band)
                    jwst_filter = 'F322W2'
                if band in ['F405N', 'F466N', 'F470N']:
                    pupil = copy.deepcopy(band)
                    jwst_filter = 'F444W'

            elif band in MIRI_BANDS:
                band_type = 'miri'
            else:
                raise Warning('Unknown band %s' % band)

            band_ext = BAND_EXTS[band_type]

            self.logger.info('-> Processing band %s' % band)

            if isinstance(self.steps, dict):
                steps = self.steps[band_type]
            else:
                steps = copy.deepcopy(self.steps)

            if isinstance(self.overwrites, dict):
                overwrites = self.overwrites[band_type]
            else:
                overwrites = copy.deepcopy(self.overwrites)

            if 'all' in overwrites:
                shutil.rmtree(base_band_dir)

            in_band_dir = None

            no_raw_data = False

            for step in steps:

                if no_raw_data:
                    continue

                if step not in ALLOWED_STEPS:
                    raise Warning('Step %s not recognised!' % step)

                if in_band_dir is None:
                    in_band_dir = os.path.join(base_band_dir, in_band_dir_dict[step])
                out_band_dir = os.path.join(base_band_dir, out_band_dir_dict[step])
                step_ext = step_ext_dict[step]

                overwrite = step in overwrites

                # Flush if we're overwriting
                if overwrite and step not in STEP_NO_DEL_DIR:
                    try:
                        shutil.rmtree(out_band_dir)
                    except FileNotFoundError:
                        pass

                if not os.path.exists(in_band_dir):
                    os.makedirs(in_band_dir)

                # Check number of start files
                n_input_files = len(glob.glob(os.path.join(in_band_dir, '*.fits')))
                if n_input_files == 0:

                    # If we haven't moved raw data, then move it now
                    if not raw_data_moved:

                        raw_files = glob.glob(
                            os.path.join(self.raw_dir,
                                         self.target,
                                         'mastDownload',
                                         'JWST',
                                         '*%s' % band_ext,
                                         '*%s_%s.fits' % (band_ext, step_ext),
                                         )
                        )

                        # Include any additionally specified observations
                        if self.target in self.extra_obs_to_include.keys():

                            extra_obs_to_include = self.extra_obs_to_include[self.target]
                            for other_target in extra_obs_to_include.keys():
                                for obs_to_include in extra_obs_to_include[other_target]:
                                    extra_files = glob.glob(
                                        os.path.join(self.raw_dir,
                                                     other_target,
                                                     'mastDownload',
                                                     'JWST',
                                                     '%s*%s' % (obs_to_include, band_ext),
                                                     '*%s_%s.fits' % (band_ext, step_ext),
                                                     )
                                    )

                                    raw_files.extend(extra_files)

                        if len(raw_files) == 0:
                            self.logger.warning('-> No raw files found. Skipping')
                            shutil.rmtree(base_band_dir)
                            no_raw_data = True
                            continue

                        raw_files.sort()

                        for raw_file in tqdm(raw_files, ascii=True, desc='Moving raw files'):

                            raw_fits_name = raw_file.split(os.path.sep)[-1]
                            hdu_out_name = os.path.join(in_band_dir, raw_fits_name)

                            # If we have a failed observation, skip it
                            skip_file = False
                            for obs in self.obs_to_skip:
                                if obs in raw_fits_name:
                                    skip_file = True
                            if skip_file:
                                continue

                            if not os.path.exists(hdu_out_name) or overwrite:

                                try:
                                    hdu = fits.open(raw_file)
                                except OSError:
                                    raise Warning('Issue with %s!' % raw_file)

                                hdu_filter = hdu[0].header['FILTER'].strip()

                                if band_type == 'nircam':

                                    hdu_pupil = hdu[0].header['PUPIL'].strip()
                                    if hdu_filter == jwst_filter and hdu_pupil == pupil:
                                        hdu.writeto(hdu_out_name, overwrite=True)

                                elif band_type == 'miri':
                                    if hdu_filter == jwst_filter:
                                        hdu.writeto(hdu_out_name, overwrite=True)

                                hdu.close()

                        raw_data_moved = True

                    else:
                        self.logger.warning('-> No files found. Skipping')
                        shutil.rmtree(base_band_dir)
                        continue

                self.logger.info('-> Doing step %s' % step)

                if step == 'lv1':

                    # Run level 1 pipeline
                    self.run_pipeline(band=band,
                                      input_dir=in_band_dir,
                                      output_dir=out_band_dir,
                                      asn_file='',
                                      pipeline_stage='lv1',
                                      overwrite=overwrite,
                                      )

                elif step == 'lv2':

                    # Run lv2 asn generation
                    asn_file = self.run_asn2(directory=in_band_dir,
                                             band=band,
                                             process_bgr_like_science=self.process_bgr_like_science,
                                             overwrite=overwrite,
                                             )

                    # Run pipeline
                    self.run_pipeline(band=band,
                                      input_dir=in_band_dir,
                                      output_dir=out_band_dir,
                                      asn_file=asn_file,
                                      pipeline_stage='lv2',
                                      overwrite=overwrite,
                                      )

                elif step == 'destripe':

                    if band_type == 'nircam':

                        cal_files = glob.glob(os.path.join(in_band_dir,
                                                           '*_%s.fits' % step_ext)
                                              )
                        cal_files.sort()

                        if len(cal_files) == 0:
                            self.logger.warning('-> No files found. Skipping')
                            shutil.rmtree(base_band_dir)
                            continue

                        cal_files.sort()

                        self.run_destripe(files=cal_files,
                                          out_dir=out_band_dir,
                                          band=band,
                                          )

                    else:

                        # Don't update the current folder
                        continue

                elif step == 'lyot_adjust':

                    if band_type == 'miri':

                        cal_files = glob.glob(os.path.join(in_band_dir,
                                                           '*_%s.fits' % step_ext)
                                              )

                        if len(cal_files) == 0:
                            self.logger.warning('-> No files found. Skipping')
                            shutil.rmtree(base_band_dir)
                            continue

                        cal_files.sort()

                        if self.lyot_method == 'adjust':
                            self.adjust_lyot(in_files=cal_files,
                                             out_dir=out_band_dir,
                                             )
                        elif self.lyot_method == 'mask':
                            self.mask_lyot(in_files=cal_files,
                                           out_dir=out_band_dir,
                                           )

                    else:

                        # Don't update the current folder
                        continue

                elif step == 'wcs_adjust':

                    if self.target in self.wcs_adjust_dict.keys():

                        cal_files = glob.glob(os.path.join(in_band_dir,
                                                           '*_%s.fits' % step_ext)
                                              )

                        cal_files.sort()

                        if len(cal_files) == 0:
                            self.logger.warning('-> No files found. Skipping')
                            shutil.rmtree(base_band_dir)
                            continue

                        self.wcs_adjust(input_dir=in_band_dir,
                                        output_dir=out_band_dir,
                                        )

                    else:

                        # Don't update the current folder
                        continue

                elif step == 'dither_match':
                    # Match the backgrounds for each set of dithers

                    cal_files = glob.glob(os.path.join(in_band_dir,
                                                       '*_%s.fits' % step_ext)
                                          )

                    cal_files.sort()

                    if len(cal_files) == 0:
                        self.logger.warning('-> No files found. Skipping')
                        shutil.rmtree(base_band_dir)
                        continue

                    if not os.path.exists(out_band_dir):
                        os.makedirs(out_band_dir)

                    out_files = glob.glob(os.path.join(out_band_dir,
                                                       '*.fits'))
                    if len(out_files) == 0 or overwrite:

                        dithers = np.unique(['_'.join(os.path.split(cal_file)[-1].split('_')[:2])
                                             for cal_file in cal_files])

                        for dither in tqdm(dithers, ascii=True):
                            dither_files = glob.glob(os.path.join(in_band_dir,
                                                                  '%s*_%s.fits' % (dither, step_ext))
                                                     )

                            dither_files.sort()
                            background_match(dither_files,
                                             out_dir=out_band_dir)

                elif step == 'bg_sub':
                    # Subtract a sigma-clipped background from each image

                    cal_files = glob.glob(os.path.join(in_band_dir,
                                                       '*_%s.fits' % step_ext)
                                          )

                    cal_files.sort()

                    if not os.path.exists(out_band_dir):
                        os.makedirs(out_band_dir)

                    if len(cal_files) == 0:
                        self.logger.warning('-> No files found. Skipping')
                        shutil.rmtree(base_band_dir)
                        continue

                    bg_sub_args = get_default_args(background_subtract)

                    for hdu_in_name in tqdm(cal_files, ascii=True):

                        hdu_out_name = os.path.join(out_band_dir, hdu_in_name.split(os.path.sep)[-1])

                        if not os.path.exists(hdu_out_name) or overwrite:

                            bg_sub_kws = {}
                            for bg_sub_arg in bg_sub_args.keys():

                                if bg_sub_arg in self.bg_sub_parameter_dict.keys():
                                    arg_val = parse_parameter_dict(self.bg_sub_parameter_dict,
                                                                   bg_sub_arg,
                                                                   band,
                                                                   self.target,
                                                                   )
                                    if arg_val == 'VAL_NOT_FOUND':
                                        arg_val = bg_sub_args[bg_sub_arg]
                                else:
                                    arg_val = bg_sub_args[bg_sub_arg]

                                bg_sub_kws[bg_sub_arg] = arg_val

                            hdu = background_subtract(hdu_in_name,
                                                      **bg_sub_kws)

                            hdu.writeto(hdu_out_name, overwrite=True)

                            hdu.close()

                elif step == 'lv3':

                    if self.use_field_in_lev3 is not None:
                        out_band_dir += '_field_' + \
                                        '_'.join(np.atleast_1d(self.use_field_in_lev3).astype(str))

                    # Run lv3 asn generation
                    asn_file = self.run_asn3(directory=in_band_dir,
                                             band=band,
                                             overwrite=overwrite)

                    # Run pipeline
                    self.run_pipeline(band=band,
                                      input_dir=in_band_dir,
                                      output_dir=out_band_dir,
                                      asn_file=asn_file,
                                      pipeline_stage='lv3',
                                      overwrite=overwrite)

                    # Also process backgrounds, if requested
                    if self.process_bgr_like_science:
                        asn_file_bgr = self.run_asn3(directory=in_band_dir,
                                                     band=band,
                                                     process_bgr_like_science=True,
                                                     overwrite=overwrite,
                                                     )
                        output_dir_bgr = os.path.join(base_band_dir, 'bgr')
                        self.run_pipeline(band=band,
                                          input_dir=in_band_dir,
                                          output_dir=output_dir_bgr,
                                          asn_file=asn_file_bgr,
                                          pipeline_stage='lv3',
                                          overwrite=overwrite,
                                          )

                elif step == 'astrometric_align':
                    
                    # if self.use_field_in_lev3 is not None:
                    #     in_band_dir += '_field_' + \
                    #                    '_'.join(np.atleast_1d(self.use_field_in_lev3).astype(str))
                    
                    if band in self.alignment_mapping.keys():
                        self.align_wcs_to_jwst(in_band_dir,
                                               band,
                                               overwrite=overwrite,
                                               )
                    else:
                        self.align_wcs_to_ref(in_band_dir,
                                              band,
                                              overwrite=overwrite,
                                              )

                else:

                    raise Warning('Step %s not recognised!' % step)

                in_band_dir = copy.deepcopy(out_band_dir)

    def run_destripe(self,
                     files,
                     out_dir,
                     band,
                     ):
        """Run destriping algorithm, looping over calibrated files

        Args:
            * files (list): List of files to loop over
            * out_dir (str): Where to save destriped files to
            * band (str): JWST band
        """

        plot_dir = os.path.join(out_dir, 'plots')
        pca_dir = os.path.join(out_dir, 'pca')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        if 'destriping_method' in self.destripe_parameter_dict.keys():
            if self.destripe_parameter_dict['destriping_method'] == 'pca':
                if not os.path.exists(pca_dir):
                    os.makedirs(pca_dir)

        with mp.get_context('fork').Pool(self.procs) as pool:

            results = []

            for result in tqdm(pool.imap(partial(parallel_destripe,
                                                 band=band,
                                                 target=self.target,
                                                 destripe_parameter_dict=self.destripe_parameter_dict,
                                                 pca_dir=pca_dir,
                                                 out_dir=out_dir,
                                                 plot_dir=plot_dir,
                                                 ),
                                         files),
                               total=len(files), ascii=True):
                results.append(result)

    def adjust_lyot(self,
                    in_files,
                    out_dir,
                    ):
        """Adjust lyot coronagraph to background level of main chip with sigma-clipped statistics

        Args:
            * in_files (list): List of files to loop over
            * out_dir (str): Where to save files to

        """

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for hdu_name in tqdm(in_files, ascii=True):
            hdu = fits.open(hdu_name)

            out_name = os.path.join(out_dir,
                                    os.path.split(hdu_name)[-1])

            if os.path.exists(out_name):
                return True

            zero_idx = np.where(hdu['SCI'].data == 0)

            # Pull out coronagraph, mask 0s and bad data quality
            lyot = copy.deepcopy(hdu['SCI'].data[735:, :290])
            lyot_dq = copy.deepcopy(hdu['DQ'].data[735:, :290])
            lyot[lyot == 0] = np.nan
            lyot[lyot_dq != 0] = np.nan

            # Pull out image, mask 0s and bad data quality
            image = copy.deepcopy(hdu['SCI'].data[:, 360:])
            image_dq = copy.deepcopy(hdu['DQ'].data[:, 360:])
            image[image == 0] = np.nan
            image[image_dq != 0] = np.nan

            # Create a mask, and do the sigma-clipped stats

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lyot_mask = make_source_mask(lyot,
                                             nsigma=3,
                                             npixels=3,
                                             )
                image_mask = make_source_mask(image,
                                              nsigma=3,
                                              npixels=3,
                                              )

                bgr_lyot = sigma_clipped_stats(lyot, mask=lyot_mask)[1]
                bgr_image = sigma_clipped_stats(image, mask=image_mask)[1]

            hdu['SCI'].data[735:, :290] += (bgr_image - bgr_lyot)
            hdu['SCI'].data[zero_idx] = 0

            hdu.writeto(out_name, overwrite=True)

            hdu.close()

    def mask_lyot(self,
                  in_files,
                  out_dir,
                  ):
        """Mask lyot coronagraph by editing DQ values

        Args:
            * in_files (list): List of files to loop over
            * out_dir (str): Where to save files to

        """

        lyot_i = slice(735, None)
        lyot_j = slice(None, 290)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for hdu_name in tqdm(in_files, ascii=True):
            hdu = fits.open(hdu_name)

            out_name = os.path.join(out_dir,
                                    os.path.split(hdu_name)[-1])

            if os.path.exists(out_name):
                return True

            hdu['SCI'].data[lyot_i, lyot_j] = np.nan
            hdu['ERR'].data[lyot_i, lyot_j] = np.nan
            hdu['DQ'].data[lyot_i, lyot_j] = 513  # Masks the coronagraph area like the other coronagraphs
            hdu.writeto(out_name, overwrite=True)

            hdu.close()

    def wcs_adjust(self,
                   input_dir,
                   output_dir,
                   ):
        """Adjust WCS so the tweakreg solution is closer to 0. Should be run on cal.fits files

        Args:
            input_dir (str): Input directory
            output_dir (str): Where to save files to
        """

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_files = glob.glob(os.path.join(input_dir,
                                             '*cal.fits')
                                )
        input_files.sort()

        repeat_keys = []

        for input_file in tqdm(input_files,
                               ascii=True):

            output_file = os.path.join(output_dir,
                                       os.path.split(input_file)[-1],
                                       )

            if os.path.exists(output_file):
                continue

            # Set up the WCSCorrector per tweakreg
            input_im = datamodels.open(input_file)

            model_name = os.path.splitext(input_im.meta.filename)[0].strip('_- ')
            refang = input_im.meta.wcsinfo.instance

            im = JWSTWCSCorrector(
                wcs=input_im.meta.wcs,
                wcsinfo={
                    'roll_ref': refang['roll_ref'],
                    'v2_ref': refang['v2_ref'],
                    'v3_ref': refang['v3_ref']
                },
                meta={
                    'image_model': input_im,
                    'name': model_name
                }
            )

            # Pull out the info we need to shift
            visit = os.path.split(input_file)[-1].split('_')[0]

            try:
                wcs_adjust_vals = self.wcs_adjust_dict[self.target][visit]

                try:
                    matrix = wcs_adjust_vals['matrix']
                except KeyError:
                    matrix = [[1, 0], [0, 1]]

                try:
                    shift = wcs_adjust_vals['shift']
                except KeyError:
                    shift = [0, 0]

            except KeyError:
                if visit not in repeat_keys:
                    self.logger.info('No WCS adjust info found for %s. Defaulting to no shift' % visit)
                    repeat_keys.append(visit)
                matrix = [[1, 0], [0, 1]]
                shift = [0, 0]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                im.set_correction(matrix=matrix, shift=shift)

                input_im.meta.wcs = im.wcs

                try:
                    update_fits_wcsinfo(
                        input_im,
                        max_pix_error=0.01,
                        npoints=16,
                    )
                except (ValueError, RuntimeError) as e:
                    self.logger.warning(
                        "Failed to update 'meta.wcsinfo' with FITS SIP "
                        f'approximation. Reported error is:\n"{e.args[0]}"'
                    )

            input_im.save(output_file)

    def run_asn2(self,
                 directory=None,
                 band=None,
                 process_bgr_like_science=False,
                 overwrite=False,
                 ):
        """Setup asn lv2 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
            * process_bgr_like_science (bool): if True, than additionally process the offset images
                in the same way as the science (for testing purposes only)
            * overwrite (bool): Whether to overwrite or not. Defaults to False.
        """

        if directory is None:
            raise Warning('Directory should be specified!')
        if band is None:
            raise Warning('Band should be specified!')

        check_bgr = True

        if band in NIRCAM_BANDS:
            band_type = 'nircam'

            # Turn off checking background for parallel off NIRCAM images:
            if self.bgr_check_type == 'parallel_off':
                check_bgr = False
        elif band in MIRI_BANDS:
            band_type = 'miri'
        else:
            raise Warning('Band %s not recognised!' % band)

        band_ext = BAND_EXTS[band_type]

        orig_dir = os.getcwd()

        os.chdir(directory)

        asn_lv2_filename = 'asn_lv2_%s.json' % band

        if not os.path.exists(asn_lv2_filename) or overwrite:

            tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', "Program"],
                        dtype=[str, str, str, str, str, float, str, str])

            all_fits_files = glob.glob('*%s_rate.fits' % band_ext)
            for f in all_fits_files:
                tab.add_row(parse_fits_to_table(f,
                                                check_bgr=check_bgr,
                                                check_type=self.bgr_check_type,
                                                background_name=self.bgr_background_name,
                                                )
                            )
            tab.sort(keys='Start')

            json_content = {"asn_type": "image2",
                            "asn_rule": "DMSLevel2bBase",
                            "version_id": time.strftime('%Y%m%dt%H%M%S'),
                            "code_version": jwst.__version__,
                            "degraded_status": "No known degraded exposures in association.",
                            "program": tab['Program'][0],
                            "constraints": "none",
                            "asn_id": 'o' + (tab['Obs_ID'][0]),
                            "asn_pool": "none",
                            "products": []
                            }

            # Loop over science first, then backgrounds
            sci_tab = tab[tab['Type'] == 'sci']
            bgr_tab = tab[tab['Type'] == 'bgr']

            for row in sci_tab:
                json_content['products'].append({
                    'name': os.path.split(row['File'])[1].split('_rate.fits')[0],
                    'members': [
                        {'expname': row['File'],
                         'exptype': 'science',
                         'exposerr': 'null'}
                    ]
                })

            # For testing purposes - enable level2 reduction for off images in the same way as the science
            if band_type in self.bgr_observation_types and process_bgr_like_science:
                for row_id, row in enumerate(bgr_tab):
                    json_content['products'].append({
                        'name': f'offset_{band}_{row_id + 1}',
                        'members': [
                            {'expname': row['File'],
                             'exptype': 'science',
                             'exposerr': 'null'}
                        ]
                    })

            # Associate background files, but only for observations with background obs
            if band_type in self.bgr_observation_types:
                for product in json_content['products']:
                    for row in bgr_tab:
                        product['members'].append({
                            'expname': row['File'],
                            'exptype': 'background',
                            'exposerr': 'null'
                        })

            with open(asn_lv2_filename, 'w') as f:
                json.dump(json_content, f)

        os.chdir(orig_dir)

        return asn_lv2_filename

    def run_asn3(self,
                 directory=None,
                 band=None,
                 process_bgr_like_science=False,
                 overwrite=False,
                 ):
        """Setup asn lv3 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
            * process_bgr_like_science (bool): Additionally process the offset images in the same way as the science
            * overwrite (bool, optional): Whether to overwrite asn file. Defaults to False.
        """

        if band is None:
            raise Warning('Band must be specified!')

        check_bgr = True

        if band in NIRCAM_BANDS:
            band_type = 'nircam'
            # Turn off checking background for parallel off NIRCAM images:
            if self.bgr_check_type == 'parallel_off':
                check_bgr = False
        elif band in MIRI_BANDS:
            band_type = 'miri'
        else:
            raise Warning('Band %s not recognised!' % band)

        band_ext = BAND_EXTS[band_type]

        orig_dir = os.getcwd()

        os.chdir(directory)

        ending = ''
        if self.use_field_in_lev3 is not None and not process_bgr_like_science:
            ending += ('_' + '_'.join(np.atleast_1d(self.use_field_in_lev3).astype(str)))
        if process_bgr_like_science:
            ending += '_offset'
        asn_lv3_filename = 'asn_lv3_%s%s.json' % (band, ending)

        if not os.path.exists(asn_lv3_filename) or overwrite:

            if not process_bgr_like_science:
                lv2_files = [f for f in glob.glob('*%s_cal.fits' % band_ext) if 'offset' not in f]
            else:
                lv2_files = [f for f in glob.glob('*_cal.fits') if 'offset' in f]

            lv2_files.sort()

            tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', 'Program'],
                        dtype=[str, str, str, str, str, float, str, str])

            for f in lv2_files:
                if self.use_field_in_lev3 is not None:
                    curfield_num = int(f.split("_")[-5][-3:])
                    if not any(np.atleast_1d(self.use_field_in_lev3) == curfield_num):
                        continue

                tab.add_row(parse_fits_to_table(f,
                                                check_bgr=check_bgr,
                                                check_type=self.bgr_check_type,
                                                background_name=self.bgr_background_name,
                                                )
                            )
            
            json_content = {"asn_type": "None",
                            "asn_rule": "DMS_Level3_Base",
                            "version_id": time.strftime('%Y%m%dt%H%M%S'),
                            "code_version": jwst.__version__,
                            "degraded_status": "No known degraded exposures in association.",
                            "program": tab['Program'][0],
                            "constraints": "No constraints",
                            "asn_id": 'o' + tab['Obs_ID'][0],
                            "asn_pool": "none",
                            "products": [{'name': '%s_%s_lv3_%s' % (self.target.lower(), band_type, band.lower()),
                                          'members': []}]
                            }

            # Make sure we're not including the MIRI backgrounds here
            if not process_bgr_like_science:
                sci_tab = tab[tab['Type'] == 'sci']
            else:
                sci_tab = tab

            for row in sci_tab:
                json_content['products'][-1]['members'].append(
                    {'expname': row['File'],
                     'exptype': 'science',
                     'exposerr': 'null'}
                )

            with open(asn_lv3_filename, 'w') as f:
                json.dump(json_content, f)

        os.chdir(orig_dir)

        return asn_lv3_filename

    def run_pipeline(self,
                     band,
                     input_dir,
                     output_dir,
                     asn_file,
                     pipeline_stage,
                     overwrite=False,
                     ):
        """Run JWST pipeline.

        Args:
            * band (str): JWST filter
            * input_dir (str): Files associated to asn_file
            * output_dir (str): Where to save the pipeline outputs
            * asn_file (str): Path to asn file. For lv1, this isn't used
            * pipeline_stage (str): Pipeline processing stage. Should be 'lv1', 'lv2', or 'lv3'
            * overwrite (bool): Whether to overwrite or not. Defaults to 'False'
        """

        if band in NIRCAM_BANDS:
            band_type = 'nircam'
        elif band in MIRI_BANDS:
            band_type = 'miri'
        else:
            raise Warning('Band %s not recognised!' % band)

        if band_type == 'nircam':
            if int(band[1:4]) <= 212:
                band_type_short_long = 'nircam_short'
            else:
                band_type_short_long = 'nircam_long'
        else:
            band_type_short_long = 'miri'

        orig_dir = os.getcwd()

        os.chdir(input_dir)

        if pipeline_stage == 'lv1':

            if overwrite:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or overwrite:

                uncal_files = glob.glob('*_uncal.fits'
                                        )

                uncal_files.sort()

                for uncal_file in uncal_files:

                    # Sometimes the WCS is catastrophically wrong. Try to correct that here
                    if self.correct_lv1_wcs:
                        if 'MAST_API_TOKEN' not in os.environ.keys():
                            os.environ['MAST_API_TOKEN'] = input('Input MAST API token: ')

                        os.system('set_telescope_pointing.py %s' % uncal_file)

                    config = calwebb_detector1.Detector1Pipeline.get_config_from_reference(uncal_file)
                    detector1 = calwebb_detector1.Detector1Pipeline.from_config_section(config)

                    # Pull out the trapsfilled file from preceding exposure if needed. Only for NIRCAM

                    persist_file = ''

                    if band_type == 'nircam':
                        uncal_file_split = uncal_file.split('_')
                        exposure_str = uncal_file_split[2]
                        exposure_int = int(exposure_str)

                        if exposure_int > 1:
                            previous_exposure_str = '%05d' % (exposure_int - 1)
                            persist_file = uncal_file.replace(exposure_str, previous_exposure_str)
                            persist_file = persist_file.replace('_uncal.fits', '_trapsfilled.fits')
                            persist_file = os.path.join(output_dir, persist_file)

                    # Specify the name of the trapsfilled file
                    detector1.persistence.input_trapsfilled = persist_file

                    # Set other parameters
                    detector1.output_dir = output_dir
                    if not os.path.isdir(detector1.output_dir):
                        os.makedirs(detector1.output_dir)

                    detector1 = attribute_setter(detector1,
                                                 parameter_dict=self.lv1_parameter_dict,
                                                 band=band,
                                                 target=self.target)

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        # Run the level 1 pipeline
                        detector1.run(uncal_file)

        elif pipeline_stage == 'lv2':

            if overwrite:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or overwrite:

                os.system('rm -rf %s' % output_dir)

                config = calwebb_image2.Image2Pipeline.get_config_from_reference(asn_file)
                im2 = calwebb_image2.Image2Pipeline.from_config_section(config)
                im2.output_dir = output_dir
                if not os.path.isdir(im2.output_dir):
                    os.makedirs(im2.output_dir)

                im2 = attribute_setter(im2,
                                       parameter_dict=self.lv2_parameter_dict,
                                       band=band,
                                       target=self.target,
                                       )

                if self.updated_flats_dir is not None:
                    my_flat = [f for f in glob.glob(os.path.join(self.updated_flats_dir, "*.fits")) if band in f]
                    if len(my_flat) != 0:
                        im2.flat_field.user_supplied_flat = my_flat[0]
                # Run the level 2 pipeline
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    im2.run(asn_file)

        elif pipeline_stage == 'lv3':

            if overwrite:
                os.system('rm -rf %s' % output_dir)

            output_fits = '%s_%s_lv3_%s_i2d.fits' % (self.target.lower(), band_type, band.lower())
            output_file = os.path.join(output_dir, output_fits)

            if not os.path.exists(output_file) or overwrite:

                os.system('rm -rf %s' % output_dir)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                # FWHM should be set per-band for both tweakreg and source catalogue
                fwhm_pix = FWHM_PIX[band]

                # Set up to run lv3 pipeline

                config = calwebb_image3.Image3Pipeline.get_config_from_reference(asn_file)
                im3 = calwebb_image3.Image3Pipeline.from_config_section(config)
                im3.output_dir = output_dir

                im3.tweakreg.kernel_fwhm = fwhm_pix * 2
                im3.source_catalog.kernel_fwhm = fwhm_pix * 2

                if band_type_short_long in self.tweakreg_create_custom_catalogs:
                    im3.tweakreg.use_custom_catalogs = True

                # Remove the tweakreg_source_catalog part, if it exists
                lv3_parameter_dict = copy.deepcopy(self.lv3_parameter_dict)
                if 'tweakreg_source_catalog' in lv3_parameter_dict.keys():
                    lv3_parameter_dict.pop('tweakreg_source_catalog')

                im3 = attribute_setter(im3,
                                       parameter_dict=lv3_parameter_dict,
                                       band=band,
                                       target=self.target,
                                       )

                # Load the asn file in, so we have access to everything we need later
                asn_file = datamodels.ModelContainer(asn_file)

                if band_type_short_long in self.tweakreg_create_custom_catalogs:
                    # Create custom catalogs for tweakreg

                    for model in asn_file._models:

                        config = SourceCatalogStep.get_config_from_reference(asn_file)
                        catalog = SourceCatalogStep.from_config_section(config)
                        catalog.output_dir = input_dir
                        catalog.kernel_fwhm = fwhm_pix * 2
                        catalog.save_results = True

                        try:
                            catalog_params = self.lv3_parameter_dict['tweakreg_source_catalog']
                        except KeyError:
                            catalog_params = {}

                        for catalog_key in catalog_params:
                            value = parse_parameter_dict(catalog_params,
                                                         catalog_key,
                                                         band,
                                                         self.target,
                                                         )

                            if value == 'VAL_NOT_FOUND':
                                continue

                            recursive_setattr(catalog, catalog_key, value)

                        edit_model = copy.deepcopy(model)

                        # Mask out bad data quality
                        dq_idx = edit_model.dq != 0
                        edit_model.data[dq_idx] = 0
                        edit_model.err[dq_idx] = 0
                        edit_model.wht[dq_idx] = 0
                        edit_model.wht[~dq_idx] = 1

                        catalog.run(edit_model)

                        # Finally, filter the catalog
                        original_catalog = model.meta.filename.replace('_cal.fits', '_cat.ecsv')
                        filter_catalog = model.meta.filename.replace('_cal.fits', '_cat_filter.ecsv')
                        cat = Table.read(original_catalog)

                        if 'filter' in catalog_params.keys():
                            for key in catalog_params['filter']:

                                if '_lower' in key:
                                    lower = catalog_params['filter'][key]
                                    upper_key = key.replace('_lower', '_upper')
                                    if upper_key in catalog_params['filter'].keys():
                                        upper = catalog_params['filter'][upper_key]
                                    else:
                                        upper = np.inf

                                    cat = cat[np.logical_and(lower <= cat[key.replace('_lower', '')],
                                                             cat[key.replace('_lower', '')] <= upper)]

                                elif '_upper' in key:
                                    upper = catalog_params['filter'][key]
                                    lower_key = key.replace('_upper', '_lower')
                                    if lower_key in catalog_params['filter'].keys():
                                        lower = catalog_params['filter'][lower_key]
                                    else:
                                        lower = -np.inf

                                    cat = cat[np.logical_and(lower <= cat[key.replace('_upper', '')],
                                                             cat[key.replace('_upper', '')] <= upper)]

                                else:
                                    cat = cat[cat[key] == catalog_params['filter'][key]]

                        cat.write(filter_catalog, format='ascii.ecsv', overwrite=True)

                        model.meta.tweakreg_catalog = filter_catalog

                # Run the tweakreg step with custom hacks if required

                config = TweakRegStep.get_config_from_reference(asn_file)
                tweakreg = TweakRegStep.from_config_section(config)
                tweakreg.output_dir = output_dir
                tweakreg.save_results = False
                tweakreg.kernel_fwhm = fwhm_pix * 2

                if band_type_short_long in self.tweakreg_create_custom_catalogs:
                    tweakreg.use_custom_catalogs = True

                try:
                    tweakreg_params = self.lv3_parameter_dict['tweakreg']
                except KeyError:
                    tweakreg_params = {}

                for tweakreg_key in tweakreg_params:
                    value = parse_parameter_dict(tweakreg_params,
                                                 tweakreg_key,
                                                 band,
                                                 self.target,
                                                 )

                    if value == 'VAL_NOT_FOUND':
                        continue

                    recursive_setattr(tweakreg, tweakreg_key, value)

                # Group up the dithers
                if band_type_short_long in self.group_tweakreg_dithers:
                    for model in asn_file._models:
                        model.meta.observation.exposure_number = '1'

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    asn_file = tweakreg.run(asn_file)

                # Make sure we skip tweakreg since we've already done it
                im3.tweakreg.skip = True

                # Degroup again to avoid potential weirdness later
                if band_type_short_long in self.group_tweakreg_dithers:
                    for i, model in enumerate(asn_file._models):
                        model.meta.observation.exposure_number = str(i)

                # Run the skymatch step with custom hacks if required
                config = SkyMatchStep.get_config_from_reference(asn_file)
                skymatch = SkyMatchStep.from_config_section(config)
                skymatch.output_dir = output_dir
                skymatch.save_results = False

                try:
                    skymatch_params = self.lv3_parameter_dict['skymatch']
                except KeyError:
                    skymatch_params = {}

                for skymatch_key in skymatch_params:
                    value = parse_parameter_dict(skymatch_params,
                                                 skymatch_key,
                                                 band,
                                                 self.target,
                                                 )

                    if value == 'VAL_NOT_FOUND':
                        continue

                    recursive_setattr(skymatch, skymatch_key, value)

                if band_type_short_long in self.group_skymatch_dithers:
                    for model in asn_file._models:
                        model.meta.observation.exposure_number = '1'

                # Alternatively, degroup (this will only do anything for nircam_short)
                elif band_type_short_long in self.degroup_skymatch_dithers:
                    for i, model in enumerate(asn_file._models):
                        model.meta.observation.exposure_number = str(i)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    asn_file = skymatch.run(asn_file)

                # Degroup again to avoid potential weirdness later
                if band_type_short_long in self.group_skymatch_dithers:
                    for i, model in enumerate(asn_file._models):
                        model.meta.observation.exposure_number = str(i)

                im3.skymatch.skip = True

                # Run the rest of the level 3 pipeline
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    im3.run(asn_file)

        else:

            raise Warning('Pipeline stage %s not recognised!' % pipeline_stage)

        os.chdir(orig_dir)

    def align_wcs_to_ref(self,
                         input_dir,
                         band,
                         overwrite=False,
                         ):
        """Align JWST image to external references. Either a table or an image

        Args:
            * input_dir (str): Directory to find files to align
            * band (str): JWST band to align
            * overwrite (bool): Whether to overwrite or not. Defaults to False
        """

        jwst_files = glob.glob(os.path.join(input_dir,
                                            '*i2d.fits'))

        if len(jwst_files) == 0:
            raise Warning('No files found to align!')

        if self.astrometric_alignment_type == 'image':
            if not self.astrometric_alignment_image:
                raise Warning('astrometric_alignment_image should be set!')

            if not os.path.exists(self.astrometric_alignment_image):
                raise Warning('Requested astrometric alignment image not found!')

            ref_hdu = fits.open(self.astrometric_alignment_image)

            ref_data = copy.deepcopy(ref_hdu[0].data)
            ref_data[ref_data == 0] = np.nan

            # Find sources in the input image

            source_cat_name = self.astrometric_alignment_image.replace('.fits', '_src_cat.fits')

            if not os.path.exists(source_cat_name) or overwrite:

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    mean, median, std = sigma_clipped_stats(ref_data, sigma=3)
                daofind = DAOStarFinder(fwhm=2.5, threshold=10 * std)
                sources = daofind(ref_data - median)
                sources.write(source_cat_name, overwrite=True)

            else:

                sources = QTable.read(source_cat_name)

            # Convert sources into a reference catalogue
            wcs_ref = HSTWCS(ref_hdu, 0)

            ref_tab = Table()
            ref_ra, ref_dec = wcs_ref.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)

            ref_tab['RA'] = ref_ra
            ref_tab['DEC'] = ref_dec

            ref_hdu.close()

        elif self.astrometric_alignment_type == 'table':

            if not self.astrometric_alignment_table:
                raise Warning('astrometric_alignment_table should be set!')

            if not os.path.exists(self.astrometric_alignment_table):
                raise Warning('Requested astrometric alignment table not found!')

            astro_table = QTable.read(self.astrometric_alignment_table, format='fits')

            if 'parallax' in astro_table.colnames:
                # This should be a GAIA query, so cut down based on whether there is a parallax measurement
                idx = np.where(~np.isnan(astro_table['parallax']))
                # This should be a GAIA query, so cut down based on whether there is good RA/Dec values
                # idx = np.where(np.logical_and(astro_table['ra_error'].value < 1,
                #                               astro_table['dec_error'].value < 1))
                astro_table = astro_table[idx]

            ref_tab = Table()

            ref_tab['RA'] = astro_table['ra']
            ref_tab['DEC'] = astro_table['dec']

            if 'xcentroid' in astro_table.colnames:
                ref_tab['xcentroid'] = astro_table['xcentroid']
                ref_tab['ycentroid'] = astro_table['ycentroid']

        else:

            raise Warning('astrometric_alignment_type should be one of image, table!')

        for jwst_file in jwst_files:

            aligned_file = jwst_file.replace('.fits', '_align.fits')
            aligned_table = aligned_file.replace('.fits', '_table.fits')

            if not os.path.exists(aligned_file) or overwrite:
                jwst_hdu = fits.open(jwst_file)

                jwst_data = copy.deepcopy(jwst_hdu['SCI'].data)
                jwst_data[jwst_data == 0] = np.nan

                # Read in the source catalogue from the pipeline

                source_cat_name = jwst_file.replace('i2d.fits', 'cat.ecsv')
                sources = Table.read(source_cat_name, format='ascii.ecsv')
                # convenience for CARTA viewing.
                sources.write(source_cat_name.replace('.ecsv', '.fits'), overwrite=True)

                # Filter out extended
                sources = sources[~sources['is_extended']]
                # Filter based on roundness and sharpness
                # sources = sources[np.logical_and(sources['sharpness'] >= 0.2,
                #                                  sources['sharpness'] <= 1.0)]
                # sources = sources[np.logical_and(sources['roundness'] >= -1.0,
                #                                  sources['roundness'] <= 1.0)]

                # Convert sources into a reference catalogue
                wcs_jwst = HSTWCS(jwst_hdu, 'SCI')
                wcs_jwst_corrector = FITSWCSCorrector(wcs_jwst)

                jwst_tab = Table()

                # Factors of 3600 to get into arcsec
                jwst_tab['TPx'] = sources['sky_centroid'].ra.value * 3600
                jwst_tab['TPy'] = sources['sky_centroid'].dec.value * 3600

                # RA/Dec should be TPx/TPy
                if 'TPx' not in ref_tab.colnames:
                    ref_tab['TPx'] = ref_tab['RA'] * 3600
                if 'TPy' not in ref_tab.colnames:
                    ref_tab['TPy'] = ref_tab['DEC'] * 3600

                # We'll also need x and y for later
                jwst_tab['x'] = sources['xcentroid']
                jwst_tab['y'] = sources['ycentroid']

                jwst_tab['ra'] = sources['sky_centroid'].ra.value
                jwst_tab['dec'] = sources['sky_centroid'].dec.value

                # Run a match
                match = XYXYMatch()
                for key in self.astrometry_parameter_dict.keys():

                    value = parse_parameter_dict(self.astrometry_parameter_dict,
                                                 key,
                                                 band,
                                                 self.target,
                                                 )
                    if value == 'VAL_NOT_FOUND':
                        continue

                    recursive_setattr(match, key, value, protected=True)

                ref_idx, jwst_idx = match(ref_tab, jwst_tab, tp_units='arcsec')

                fit_wcs_args = get_default_args(fit_wcs)

                fit_wcs_kws = {}
                for fit_wcs_arg in fit_wcs_args.keys():

                    if fit_wcs_arg in self.astrometry_parameter_dict.keys():
                        arg_val = parse_parameter_dict(self.astrometry_parameter_dict,
                                                       fit_wcs_arg,
                                                       band,
                                                       self.target,
                                                       )
                        if arg_val == 'VAL_NOT_FOUND':
                            arg_val = fit_wcs_args[fit_wcs_arg]
                    else:
                        arg_val = fit_wcs_args[fit_wcs_arg]

                    # sigma here is fiddly, test if it's a tuple and fix to rmse if not
                    if fit_wcs_arg == 'sigma':
                        if type(arg_val) != tuple:
                            arg_val = (arg_val, 'rmse')

                    fit_wcs_kws[fit_wcs_arg] = arg_val

                # Do alignment
                wcs_aligned_fit = fit_wcs(refcat=ref_tab[ref_idx],
                                          imcat=jwst_tab[jwst_idx],
                                          corrector=wcs_jwst_corrector,
                                          **fit_wcs_kws,
                                          )

                wcs_aligned = wcs_aligned_fit.wcs

                self.logger.info('Original WCS:')
                self.logger.info(wcs_jwst)
                self.logger.info('Updated WCS:')
                self.logger.info(wcs_aligned)

                updatehdr.update_wcs(jwst_hdu,
                                     'SCI',
                                     wcs_aligned,
                                     wcsname='TWEAK',
                                     reusename=True)

                # Also apply this to each individual crf file

                matrix = wcs_aligned_fit.meta['fit_info']['matrix']
                shift = wcs_aligned_fit.meta['fit_info']['shift']
                crf_files = glob.glob(os.path.join(input_dir,
                                                   '*_crf.fits')
                                      )

                crf_files.sort()

                with mp.get_context('fork').Pool(self.procs) as pool:

                    results = []

                    for result in tqdm(pool.imap(partial(parallel_tweakback,
                                                         matrix=matrix,
                                                         shift=shift,
                                                         ref_tpwcs=wcs_jwst_corrector,
                                                         ),
                                                 crf_files),
                                       total=len(crf_files),
                                       ascii=True,
                                       desc='tweakback',
                                       ):
                        results.append(result)

                if not all(results):
                    self.logger.warning('Not all crf files tweakbacked. May cause issues!')

                fit_info = wcs_aligned_fit.meta['fit_info']
                fit_mask = fit_info['fitmask']

                # Pull out useful alignment info to the table -- HST x/y/RA/Dec, JWST x/y/RA/Dec (corrected and
                # uncorrected)
                aligned_tab = Table()

                # Catch if there's only RA/Dec in the reference table
                if 'xcentroid' in ref_tab.colnames:
                    aligned_tab['xcentroid_ref'] = ref_tab[ref_idx]['xcentroid'][fit_mask]
                    aligned_tab['ycentroid_ref'] = ref_tab[ref_idx]['ycentroid'][fit_mask]
                aligned_tab['ra_ref'] = ref_tab[ref_idx]['RA'][fit_mask]
                aligned_tab['dec_ref'] = ref_tab[ref_idx]['DEC'][fit_mask]

                # Since we're pulling from the source catalogue, these should all exist
                aligned_tab['xcentroid_jwst'] = jwst_tab[jwst_idx]['x'][fit_mask]
                aligned_tab['ycentroid_jwst'] = jwst_tab[jwst_idx]['y'][fit_mask]
                aligned_tab['ra_jwst_uncorr'] = jwst_tab[jwst_idx]['ra'][fit_mask]
                aligned_tab['dec_jwst_uncorr'] = jwst_tab[jwst_idx]['dec'][fit_mask]

                aligned_tab['ra_jwst_corr'] = fit_info['fit_RA']
                aligned_tab['dec_jwst_corr'] = fit_info['fit_DEC']

                aligned_tab.write(aligned_table, format='fits', overwrite=True)

                jwst_hdu.writeto(aligned_file, overwrite=True)

                jwst_hdu.close()

    def align_wcs_to_jwst(self,
                          input_dir,
                          band,
                          overwrite=False,
                          ):
        """Internally align image to already aligned JWST one via cross-correlation

        Args:
            * input_dir (str): Directory to find files to align
            * band (str): JWST band to align
            * overwrite (bool): Whether to overwrite or not. Defaults to False
        """

        jwst_files = glob.glob(os.path.join(input_dir,
                                            '*i2d.fits'))

        if len(jwst_files) == 0:
            raise Warning('No files found to align!')

        ref_band = self.alignment_mapping[band]

        if ref_band in NIRCAM_BANDS:
            ref_band_type = 'nircam'
        elif band in MIRI_BANDS:
            ref_band_type = 'miri'
        else:
            raise Warning('Reference band %s not recognised!' % band)

        if self.use_field_in_lev3 is not None:
            ref_dir = 'lv3'  # _field_' + '_'.join(np.atleast_1d(self.use_field_in_lev3).astype(str))
            ref_band = band
            if ref_band in NIRCAM_BANDS:
                ref_band_type = 'nircam'
            elif band in MIRI_BANDS:
                ref_band_type = 'miri'
        else:
            ref_dir = 'lv3'

        ref_hdu_name = os.path.join(self.reprocess_dir,
                                    self.target,
                                    ref_band,
                                    ref_dir,
                                    '%s_%s_lv3_%s_i2d_align.fits' % (self.target, ref_band_type, ref_band.lower()))

        if not os.path.exists(ref_hdu_name):
            self.logger.warning('reference HDU to align not found. Will just rename files')

        for jwst_file in jwst_files:

            aligned_file = jwst_file.replace('.fits', '_align.fits')

            if not os.path.exists(ref_hdu_name):
                if not os.path.exists(aligned_file) or overwrite:
                    os.system('cp %s %s' % (jwst_file, aligned_file))
                    continue

            if not os.path.exists(aligned_file) or overwrite:
                ref_hdu = fits.open(ref_hdu_name)
                jwst_hdu = fits.open(jwst_file)

                wcs_jwst = HSTWCS(jwst_hdu, 'SCI')
                wcs_jwst_corrector = FITSWCSCorrector(wcs_jwst)

                ref_data = copy.deepcopy(ref_hdu['SCI'].data)
                jwst_data = copy.deepcopy(jwst_hdu['SCI'].data)

                ref_err = copy.deepcopy(ref_hdu['ERR'].data)
                jwst_err = copy.deepcopy(jwst_hdu['ERR'].data)

                ref_data[ref_data == 0] = np.nan
                jwst_data[jwst_data == 0] = np.nan

                # Reproject the ref HDU to the image to align

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ref_data = reproject_interp(fits.PrimaryHDU(data=ref_data, header=ref_hdu['SCI'].header),
                                                wcs_jwst,
                                                shape_out=jwst_hdu['SCI'].data.shape,
                                                return_footprint=False,
                                                )

                    ref_err = reproject_interp(fits.PrimaryHDU(data=ref_err, header=ref_hdu['SCI'].header),
                                               wcs_jwst,
                                               shape_out=jwst_hdu['SCI'].data.shape,
                                               return_footprint=False,
                                               )

                nan_idx = np.logical_or(np.isnan(ref_data),
                                        np.isnan(jwst_data))

                ref_data[nan_idx] = np.nan
                jwst_data[nan_idx] = np.nan

                # ref_data[:,520:920] = np.nan
                # jwst_data[:, 520:920] = np.nan
                # ref_err[:, 520:920] = np.nan
                # jwst_err[:, 520:920] = np.nan
                #
                # ref_err[(ref_data > 100) | (ref_data<-0.1)] = np.nan
                # jwst_err[(jwst_data > 100) | (jwst_data<-0.1)] = np.nan
                # jwst_data[(jwst_data>100) | (jwst_data<-0.1)] = np.nan
                # ref_data[(ref_data > 100) | (ref_data<-0.1)] = np.nan

                ref_err[nan_idx] = np.nan
                jwst_err[nan_idx] = np.nan

                # Make sure we're square, since apparently this causes weirdness
                data_size_min = min(jwst_data.shape)
                data_slice_i = slice(jwst_data.shape[0] // 2 - data_size_min // 2,
                                     jwst_data.shape[0] // 2 + data_size_min // 2)
                data_slice_j = slice(jwst_data.shape[1] // 2 - data_size_min // 2,
                                     jwst_data.shape[1] // 2 + data_size_min // 2)

                x_off, y_off = cross_correlation_shifts(ref_data[data_slice_i, data_slice_j],
                                                        jwst_data[data_slice_i, data_slice_j],
                                                        errim1=ref_err[data_slice_i, data_slice_j],
                                                        errim2=jwst_err[data_slice_i, data_slice_j],
                                                        )

                self.logger.info('Found offset of [%.2f, %.2f]' % (x_off, y_off))

                # Apply correction directly to CRPIX
                wcs_jwst_corrector.wcs.wcs.crpix += [x_off, y_off]

                wcs_aligned = wcs_jwst_corrector.wcs

                updatehdr.update_wcs(jwst_hdu,
                                     'SCI',
                                     wcs_aligned,
                                     wcsname='TWEAK',
                                     reusename=True)

                # Also apply this to each individual crf file

                crf_files = glob.glob(os.path.join(input_dir,
                                                   '*_crf.fits')
                                      )

                crf_files.sort()

                shift = [-x_off, -y_off]

                with mp.get_context('fork').Pool(self.procs) as pool:

                    results = []

                    for result in tqdm(pool.imap(partial(parallel_tweakback,
                                                         shift=shift,
                                                         ref_tpwcs=wcs_jwst_corrector,
                                                         ),
                                                 crf_files),
                                       total=len(crf_files),
                                       ascii=True,
                                       desc='tweakback',
                                       ):
                        results.append(result)

                if not all(results):
                    self.logger.warning('Not all crf files tweakbacked. May cause issues!')

                jwst_hdu.writeto(aligned_file, overwrite=True)

                jwst_hdu.close()

                ref_hdu.close()
