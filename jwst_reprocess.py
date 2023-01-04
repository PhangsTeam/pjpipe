import copy
import functools
import glob
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
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, QTable
from drizzlepac import updatehdr
from image_registration import cross_correlation_shifts
from photutils import make_source_mask
from photutils.detection import DAOStarFinder
from reproject import reproject_interp
from stwcs.wcsutil import HSTWCS
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from tweakwcs import fit_wcs, XYXYMatch, FITSWCS
from tweakwcs.correctors import FITSWCSCorrector, JWSTWCSCorrector

from nircam_destriping import NircamDestriper

jwst = None
datamodels = None
update_fits_wcsinfo = None
calwebb_detector1 = None
calwebb_image2 = None
calwebb_image3 = None
TweakRegStep = None

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
                                     band)
        if value == 'VAL_NOT_FOUND':
            continue

        recursive_setattr(nc_destripe, key, value)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with threadpool_limits(limits=1, user_api='blas'):
            nc_destripe.run_destriping()

    return True


def parse_fits_to_table(file,
                        check_bgr=False,
                        check_type='parallel_off'
                        ):
    """Pull necessary info out of fits headers

    Args:
        file (str): File to get info for
        check_bgr (bool): Whether to check if this is a science or background observation (in the MIRI case)
        check_type (str): How to check if background observation. Options are 'parallel_off', which will use the
            filename to see if it's a parallel observation with NIRCAM, or 'off_in_name', which will use the observation
            name to check
    """

    if check_bgr:

        f_type = 'sci'

        if check_type == 'parallel_off':

            file_split = os.path.split(file)[-1]

            if file_split.split('_')[1][2] == '2':
                f_type = 'bgr'

        elif check_type == 'off_in_name':
            hdu = fits.open(file)
            if 'off' in hdu[0].header['TARGPROP'].lower():
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
                         band):
    """Pull values out of a parameter dictionary

    Args:
        parameter_dict (dict): Dictionary of parameters and associated values
        key (str): Particular key in parameter_dict to consider
        band (str): JWST band, to parse out band type and potentially per-band
            values

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
        raise Warning('Band type %s not known!')

    if isinstance(value, dict):

        if band_type in value.keys():
            value = value[band_type]

        elif band_type == 'nircam' and short_long in value.keys():
            value = value[short_long]

        elif band in value.keys():
            value = value[band]

        else:
            value = 'VAL_NOT_FOUND'

    # Finally, if we have a string with a 'pix' in there, we need to convert to arcsec
    if isinstance(value, str):
        if 'pix' in value:
            value = float(value.strip('pix')) * pixel_scale

    return value


def recursive_setattr(f, attribute, value):
    pre, _, post = attribute.rpartition('.')
    return setattr(recursive_getattr(f, pre) if pre else f, post, value)


def recursive_getattr(f, attribute, *args):
    def _getattr(f, attribute):
        return getattr(f, attribute, *args)

    return functools.reduce(_getattr, [f] + attribute.split('.'))


def attribute_setter(pipeobj, parameter_dict, band):
    for key in parameter_dict.keys():
        if type(parameter_dict[key]) is dict:
            for subkey in parameter_dict[key]:                            
                value = parse_parameter_dict(parameter_dict[key],
                                             subkey, band)
                if value == 'VAL_NOT_FOUND':
                    continue
                recursive_setattr(pipeobj, '.'.join([key, subkey]), value)
        else:
            value = parse_parameter_dict(parameter_dict,
                                         key, band)
            if value == 'VAL_NOT_FOUND':
                continue

            recursive_setattr(pipeobj, key, value)
    return(pipeobj)


class JWSTReprocess:

    def __init__(self,
                 galaxy,
                 raw_dir,
                 reprocess_dir,
                 crds_dir,
                 bands=None,
                 lv1_parameter_dict='phangs',
                 lv2_parameter_dict='phangs',
                 lv3_parameter_dict='phangs',
                 destripe_parameter_dict='phangs',
                 degroup_short_nircam=True,
                 bgr_check_type='parallel_off',
                 astrometric_alignment_type='image',
                 astrometric_alignment_image=None,
                 astrometric_alignment_table=None,
                 alignment_mapping=None,
                 wcs_adjust_dict=None,
                 tpmatch_searchrad=10,
                 tpmatch_separation=0.000001,
                 tpmatch_tolerance=0.7,
                 tpmatch_use2dhist=True,
                 tpmatch_fitgeom='shift',
                 tpmatch_nclip=3,
                 tpmatch_sigma=3,
                 do_all=True,
                 do_lv1=False,
                 do_lv2=False,
                 do_destriping=False,
                 do_lyot_adjust=None,
                 do_wcs_adjust=False,
                 do_lv3=False,
                 do_astrometric_alignment=False,
                 overwrite_all=False,
                 overwrite_lv1=False,
                 overwrite_lv2=False,
                 overwrite_destriping=False,
                 overwrite_lyot_adjust=False,
                 overwrite_wcs_adjust=False,
                 overwrite_lv3=False,
                 overwrite_astrometric_alignment=False,
                 overwrite_astrometric_ref_cat=False,
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
            * galaxy (str): Galaxy to run reprocessing for
            * raw_dir (str): Path to raw data
            * reprocess_dir (str): Path to reprocess data into
            * crds_dir (str): Path to CRDS data
            * bands (list): JWST filters to loop over
            * lv1_parameter_dict (dict): Dictionary of parameters to feed to level 1 pipeline. See description above
                for how this should be formatted. Defaults to 'phangs', which will use the parameters for the
                PHANGS-JWST reduction. To keep pipeline default, use 'None'
            * lv2_parameter_dict (dict): As `lv1_parameter_dict`, but for the level 2 pipeline
            * lv3_parameter_dict (dict): As `lv1_parameter_dict`, but for the level 3 pipeline
            * destripe_parameter_dict (dict): As `lv1_parameter_dict`, but for the destriping procedure
            * degroup_short_nircam (bool): Will degroup short wavelength NIRCAM observations for all steps beyond
                relative alignment. This can alleviate steps between mosaic pointings
            * bgr_check_type (str): Method to check if MIRI obs is science or background. Options are 'parallel_off' and
                off_in_name. Defaults to 'parallel_off'
            * astrometric_alignment_image (str): Path to image to align astrometry to
            * astrometric_alignment_table (str): Path to table to align astrometry to
            * alignment_mapping (dict): Dictionary to map basing alignments off cross-correlation with other aligned
                band. Should be of the form {'band': 'reference_band'}
            * wcs_adjust_dict (dict): dict to adjust image group WCS before tweakreg step. Should be of form
                {filter: {'group': {'matrix': [[1, 0], [0, 1]], 'shift': [dx, dy]}}}. Defaults to None.
            * tpmatch_searchrad (float): Distance to search for a match when astrometric aligning. Defaults to 10
            * tpmatch_separation (float): Separation for objects to be considered separate in astrometric alignment.
                Defaults to 0.000001
            * tpmatch_tolerance (float): Max tolerance for astrometric alignment match. Defaults to 0.7
            * tpmatch_use2dhist (bool): Whether to use 2D histogram to get initial astrometric alignment offsets.
                Defaults to True.
            * tpmatch_fitgeom (str): Type of fit to do in astrometric alignment. Defaults to 'shift'
            * tpmatch_nclip (int): Number of iterations to clip in astrometric alignment matching. Defaults to 3
            * tpmatch_sigma (float): Sigma-limit for clipping in astrometric alignment. Defaults to 3
            * do_all (bool): Do all processing steps. Defaults to True
            * do_lv1 (bool): Run lv1 pipeline. Defaults to False
            * do_lv2 (bool): Run lv2 pipeline. Defaults to False
            * do_destriping (bool): Run destriping algorithm on lv2 data. Defaults to False
            * do_lyot_adjust (str): How to deal with the MIRI coronagraph. Options are 'mask', which masks it out, or
                'adjust', which will adjust the background level to match the main array
            * do_wcs_adjust (bool): Whether to run WCS adjustment before tweakreg
            * do_lv3 (bool): Run lv3 pipeline. Defaults to False
            * do_astrometric_alignment (bool): Run astrometric alignment on lv3 data. Defaults to False
            * overwrite_all (bool): Whether to overwrite everything. Defaults to False
            * overwrite_lv1 (bool): Whether to overwrite lv1 data. Defaults to False
            * overwrite_lv2 (bool): Whether to overwrite lv2 data. Defaults to False
            * overwrite_destriping (bool): Whether to overwrite destriped data. Defaults to False
            * overwrite_lyot_adjust (bool): Whether to overwrite MIRI coronagraph edits. Defaults to False
            * overwrite_wcs_adjust (bool): Whether to overwrite initial WCS adjustments. Defaults to False
            * overwrite_lv3 (bool): Whether to overwrite lv3 data. Defaults to False
            * overwrite_astrometric_alignment (bool): Whether to overwrite astrometric alignment. Defaults to False
            * overwrite_astrometric_ref_cat (bool): Whether to overwrite the generated reference catalogue for
                astrometric alignment. Defaults to False
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
            * Update destriping algorithm as we improve it
            * Record alignment parameters into the fits header

        """

        os.environ['CRDS_SERVER_URL'] = crds_url
        os.environ['CRDS_PATH'] = crds_dir

        # Use global variables, so we can import JWST stuff preserving environment variables
        global jwst
        global calwebb_detector1, calwebb_image2, calwebb_image3
        global TweakRegStep
        global datamodels, update_fits_wcsinfo

        import jwst
        from jwst import datamodels
        from jwst.assign_wcs.util import update_fits_wcsinfo
        from jwst.pipeline import calwebb_detector1, calwebb_image2, calwebb_image3
        from jwst.tweakreg import TweakRegStep

        self.galaxy = galaxy

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

        if destripe_parameter_dict is None:
            destripe_parameter_dict = {}
        elif destripe_parameter_dict == 'phangs':
            destripe_parameter_dict = {
                'quadrants': True,
                'destriping_method': 'pca',
                'dilate_size': 7,
                'pca_reconstruct_components': 5,
                'pca_diffuse': True,
                'pca_final_med_row_subtraction': False,
            }

            # Old version, using median filter
            # destripe_parameter_dict = {
            #     'quadrants': False,
            #     'destriping_method': 'median_filter',
            #     'dilate_size': 7,
            #     'median_filter_scales': [7, 31, 63, 127, 511]
            # }

        self.destripe_parameter_dict = destripe_parameter_dict

        self.degroup_short_nircam = degroup_short_nircam

        if do_all:
            do_lv1 = True
            do_lv2 = True
            do_destriping = True
            do_lv3 = True
            do_astrometric_alignment = True

        self.do_lv1 = do_lv1
        self.do_lv2 = do_lv2
        self.do_destriping = do_destriping
        self.do_lyot_adjust = do_lyot_adjust
        self.do_wcs_adjust = do_wcs_adjust
        self.do_lv3 = do_lv3
        self.do_astrometric_alignment = do_astrometric_alignment

        self.astrometric_alignment_type = astrometric_alignment_type
        self.astrometric_alignment_image = astrometric_alignment_image
        self.astrometric_alignment_table = astrometric_alignment_table

        self.tpmatch_searchrad = tpmatch_searchrad
        self.tpmatch_separation = tpmatch_separation
        self.tpmatch_tolerance = tpmatch_tolerance
        self.tpmatch_use2dhist = tpmatch_use2dhist
        self.tpmatch_fitgeom = tpmatch_fitgeom
        self.tpmatch_nclip = tpmatch_nclip
        self.tpmatch_sigma = tpmatch_sigma

        self.bgr_check_type = bgr_check_type

        if overwrite_all:
            overwrite_lv1 = True
            overwrite_lv2 = True
            overwrite_destriping = True
            overwrite_lyot_adjust = True
            overwrite_wcs_adjust = True
            overwrite_lv3 = True
            overwrite_astrometric_alignment = True

        self.overwrite_all = overwrite_all
        self.overwrite_lv1 = overwrite_lv1
        self.overwrite_lv2 = overwrite_lv2
        self.overwrite_destriping = overwrite_destriping
        self.overwrite_lyot_adjust = overwrite_lyot_adjust
        self.overwrite_wcs_adjust = overwrite_wcs_adjust
        self.overwrite_lv3 = overwrite_lv3
        self.overwrite_astrometric_alignment = overwrite_astrometric_alignment
        self.overwrite_astrometric_ref_cat = overwrite_astrometric_ref_cat

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

        self.logger.info('Reprocessing %s' % self.galaxy)

        for band in self.bands:

            pupil = 'CLEAR'
            jwst_filter = copy.deepcopy(band)

            if band in NIRCAM_BANDS:
                band_type = 'nircam'
                do_destriping = self.do_destriping
                do_lyot_adjust = None

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
                do_destriping = False
                do_lyot_adjust = self.do_lyot_adjust
            else:
                raise Warning('Unknown band %s' % band)

            band_ext = BAND_EXTS[band_type]

            self.logger.info('-> %s' % band)

            if self.overwrite_all:
                os.system('rm -rf %s' % os.path.join(self.reprocess_dir,
                                                     self.galaxy,
                                                     band))

            if self.do_lv1:

                self.logger.info('-> Level 1')

                uncal_dir = os.path.join(self.reprocess_dir,
                                         self.galaxy,
                                         band,
                                         'uncal'
                                         )

                if self.overwrite_lv1:
                    os.system('rm -rf %s' % uncal_dir)

                if not os.path.exists(uncal_dir):
                    os.makedirs(uncal_dir)

                if len(glob.glob(os.path.join(uncal_dir, '*.fits'))) == 0 or self.overwrite_lv1:

                    uncal_files = glob.glob(os.path.join(self.raw_dir,
                                                         self.galaxy,
                                                         'mastDownload',
                                                         'JWST',
                                                         '*%s' % band_ext,
                                                         '*%s_uncal.fits' % band_ext)
                                            )

                    if len(uncal_files) == 0:
                        self.logger.info('-> No uncal files found. Skipping')
                        os.system('rm -rf %s' % uncal_dir)
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                    uncal_files.sort()

                    for uncal_file in tqdm(uncal_files, ascii=True):

                        uncal_fits_name = uncal_file.split(os.path.sep)[-1]
                        hdu_out_name = os.path.join(uncal_dir, uncal_fits_name)

                        if not os.path.exists(hdu_out_name) or self.overwrite_lv1:

                            try:
                                hdu = fits.open(uncal_file)
                            except OSError:
                                raise Warning('Issue with %s!' % uncal_file)

                            if band_type == 'nircam':
                                if hdu[0].header['FILTER'].strip() == jwst_filter and \
                                        hdu[0].header['PUPIL'].strip() == pupil:
                                    hdu.writeto(hdu_out_name, overwrite=True)
                            elif band_type == 'miri':
                                if hdu[0].header['FILTER'].strip() == jwst_filter:
                                    hdu.writeto(hdu_out_name, overwrite=True)

                            hdu.close()

                if len(glob.glob(os.path.join(uncal_dir, '*.fits'))) == 0:
                    self.logger.info('-> No uncal files found. Skipping')
                    os.system('rm -rf %s' % uncal_dir)
                    shutil.rmtree(os.path.join(self.reprocess_dir,
                                               self.galaxy,
                                               band))
                    continue

                rate_dir = os.path.join(self.reprocess_dir,
                                        self.galaxy,
                                        band,
                                        'rate')

                if self.overwrite_lv1:
                    os.system('rm -rf %s' % rate_dir)

                # Run pipeline
                self.run_pipeline(band=band,
                                  input_dir=uncal_dir,
                                  output_dir=rate_dir,
                                  asn_file=None,
                                  pipeline_stage='lv1')

            if self.do_lv2:

                self.logger.info('-> Level 2')

                # Move lv2 rate files
                rate_dir = os.path.join(self.reprocess_dir,
                                        self.galaxy,
                                        band,
                                        'rate')

                cal_dir = os.path.join(self.reprocess_dir,
                                       self.galaxy,
                                       band,
                                       'cal')

                if not os.path.exists(rate_dir):
                    os.makedirs(rate_dir)

                if not self.do_lv1:

                    rate_files = glob.glob(os.path.join(self.raw_dir,
                                                        self.galaxy,
                                                        'mastDownload',
                                                        'JWST',
                                                        '*%s' % band_ext,
                                                        '*%s_rate.fits' % band_ext)
                                           )

                    if len(rate_files) == 0:
                        self.logger.info('-> No rate files found. Skipping')
                        os.system('rm -rf %s' % rate_dir)
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                    for rate_file in tqdm(rate_files, ascii=True):

                        fits_name = rate_file.split(os.path.sep)[-1]
                        if not os.path.exists(os.path.join(rate_dir, fits_name)) or self.overwrite_lv2:

                            hdu = fits.open(rate_file)
                            if band_type == 'nircam':
                                if hdu[0].header['FILTER'].strip() == jwst_filter and \
                                        hdu[0].header['PUPIL'].strip() == pupil:
                                    os.system('cp %s %s/' % (rate_file, rate_dir))
                            elif band_type == 'miri':
                                if hdu[0].header['FILTER'].strip() == jwst_filter:
                                    os.system('cp %s %s/' % (rate_file, rate_dir))

                            hdu.close()

                # Run lv2 asn generation
                asn_file = self.run_asn2(directory=rate_dir,
                                         band=band, process_bgr_like_science=self.process_bgr_like_science)

                # Run pipeline
                self.run_pipeline(band=band,
                                  input_dir=rate_dir,
                                  output_dir=cal_dir,
                                  asn_file=asn_file,
                                  pipeline_stage='lv2')

            # For now, we only destripe the NIRCAM data
            if do_destriping:

                self.logger.info('-> Destriping')

                destripe_dir = os.path.join(self.reprocess_dir,
                                            self.galaxy,
                                            band,
                                            'destripe')

                if self.overwrite_destriping:
                    os.system('rm -rf %s' % destripe_dir)

                if not os.path.exists(destripe_dir):
                    os.makedirs(destripe_dir)

                if self.do_lv2:
                    cal_files = glob.glob(os.path.join(self.reprocess_dir,
                                                       self.galaxy,
                                                       band,
                                                       'cal',
                                                       '*%s_cal.fits' % band_ext)
                                          )

                    if len(cal_files) == 0:
                        self.logger.info('-> No cal files found. Skipping')
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                else:
                    initial_cal_files = glob.glob(os.path.join(self.raw_dir,
                                                               self.galaxy,
                                                               'mastDownload',
                                                               'JWST',
                                                               '*%s' % band_ext,
                                                               '*%s_cal.fits' % band_ext)
                                                  )

                    if len(initial_cal_files) == 0:
                        self.logger.info('-> No cal files found. Skipping')
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                    cal_files = []

                    for cal_file in initial_cal_files:

                        hdu = fits.open(cal_file)
                        if band_type == 'nircam':
                            if hdu[0].header['FILTER'].strip() == jwst_filter and \
                                    hdu[0].header['PUPIL'].strip() == pupil:
                                cal_files.append(cal_file)
                        elif band_type == 'miri':
                            if hdu[0].header['FILTER'].strip() == jwst_filter:
                                cal_files.append(cal_file)
                        hdu.close()

                cal_files.sort()
                self.run_destripe(files=cal_files,
                                  out_dir=destripe_dir,
                                  band=band,
                                  )

            # The Lyot coronagraph is only in the MIRI bands
            if do_lyot_adjust is not None:

                self.logger.info('-> Adjusting lyot with method %s' % do_lyot_adjust)

                lyot_adjust_dir = os.path.join(self.reprocess_dir,
                                               self.galaxy,
                                               band,
                                               'lyot_adjust')

                if self.overwrite_lyot_adjust:
                    os.system('rm -rf %s' % lyot_adjust_dir)

                if not os.path.exists(lyot_adjust_dir):
                    os.makedirs(lyot_adjust_dir)

                if self.do_lv2:
                    cal_files = glob.glob(os.path.join(self.reprocess_dir,
                                                       self.galaxy,
                                                       band,
                                                       'cal',
                                                       '*%s_cal.fits' % band_ext)
                                          )

                    if len(cal_files) == 0:
                        self.logger.info('-> No cal files found. Skipping')
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                else:
                    initial_cal_files = glob.glob(os.path.join(self.raw_dir,
                                                               self.galaxy,
                                                               'mastDownload',
                                                               'JWST',
                                                               '*%s' % band_ext,
                                                               '*%s_cal.fits' % band_ext)
                                                  )

                    if len(initial_cal_files) == 0:
                        self.logger.info('-> No cal files found. Skipping')
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                    cal_files = []

                    for cal_file in initial_cal_files:

                        hdu = fits.open(cal_file)
                        if band_type == 'nircam':
                            if hdu[0].header['FILTER'].strip() == jwst_filter and \
                                    hdu[0].header['PUPIL'].strip() == pupil:
                                cal_files.append(cal_file)
                        elif band_type == 'miri':
                            if hdu[0].header['FILTER'].strip() == jwst_filter:
                                cal_files.append(cal_file)
                        hdu.close()

                cal_files.sort()

                if do_lyot_adjust == 'adjust':
                    self.adjust_lyot(in_files=cal_files,
                                     out_dir=lyot_adjust_dir)
                elif do_lyot_adjust == 'mask':
                    self.mask_lyot(in_files=cal_files,
                                   out_dir=lyot_adjust_dir)

            if self.do_wcs_adjust and band in self.wcs_adjust_dict.keys():

                self.logger.info('-> Adjusting WCS')

                if self.do_lv2 and not (do_destriping or do_lyot_adjust is not None):
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'cal')
                elif do_lyot_adjust is not None:
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'lyot_adjust')
                elif do_destriping:
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'destripe')
                else:
                    # At this point, we pull the cal files from the raw directories
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'lv2')

                    if not os.path.exists(input_dir):
                        os.makedirs(input_dir)

                    cal_files = [f for f in glob.glob(os.path.join(self.raw_dir,
                                                                   self.galaxy,
                                                                   'mastDownload',
                                                                   'JWST',
                                                                   '*%s' % band_ext,
                                                                   '*%s_cal.fits' % band_ext)) if ('offset' not in f)]

                    if len(cal_files) == 0:
                        self.logger.info('-> No cal files found. Skipping')
                        os.system('rm -rf %s' % input_dir)
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                    for cal_file in tqdm(cal_files, ascii=True):

                        fits_name = cal_file.split(os.path.sep)[-1]
                        if os.path.exists(os.path.join(input_dir, fits_name)) or self.overwrite_lv3:
                            continue

                        hdu = fits.open(cal_file)
                        if band_type == 'nircam':
                            if hdu[0].header['FILTER'].strip() == jwst_filter and \
                                    hdu[0].header['PUPIL'].strip() == pupil:
                                os.system('cp %s %s/' % (cal_file, input_dir))
                        elif band_type == 'miri':
                            if hdu[0].header['FILTER'].strip() == jwst_filter:
                                os.system('cp %s %s/' % (cal_file, input_dir))
                        hdu.close()

                output_dir = os.path.join(self.reprocess_dir,
                                          self.galaxy,
                                          band,
                                          'wcs_adjust')

                if self.overwrite_wcs_adjust:
                    os.system('rm -rf %s' % output_dir)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                self.wcs_adjust(input_dir=input_dir,
                                output_dir=output_dir,
                                band=band)

            if self.do_lv3:

                self.logger.info('-> Level 3')

                if self.use_field_in_lev3 is None:

                    output_dir = os.path.join(self.reprocess_dir,
                                              self.galaxy,
                                              band,
                                              'lv3')

                else:
                    output_dir = os.path.join(self.reprocess_dir,
                                              self.galaxy, band,
                                              'lv3_field_' + '_'.join(np.atleast_1d(
                                                  self.use_field_in_lev3).astype(str)))

                if self.overwrite_lv3:
                    os.system('rm -rf %s' % output_dir)

                if self.wcs_adjust_dict and band in self.wcs_adjust_dict.keys():
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'wcs_adjust')
                elif self.do_lv2 and not (do_destriping or do_lyot_adjust is not None):
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'cal')
                elif do_lyot_adjust is not None:
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'lyot_adjust')
                elif do_destriping:
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'destripe')
                else:
                    # At this point, we pull the cal files from the raw directories
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'lv2')

                    if not os.path.exists(input_dir):
                        os.makedirs(input_dir)

                    cal_files = [f for f in glob.glob(os.path.join(self.raw_dir,
                                                                   self.galaxy,
                                                                   'mastDownload',
                                                                   'JWST',
                                                                   '*%s' % band_ext,
                                                                   '*%s_cal.fits' % band_ext)) if ('offset' not in f)]

                    if len(cal_files) == 0:
                        self.logger.info('-> No cal files found. Skipping')
                        os.system('rm -rf %s' % input_dir)
                        shutil.rmtree(os.path.join(self.reprocess_dir,
                                                   self.galaxy,
                                                   band))
                        continue

                    for cal_file in tqdm(cal_files, ascii=True):

                        fits_name = cal_file.split(os.path.sep)[-1]
                        if os.path.exists(os.path.join(input_dir, fits_name)) or self.overwrite_lv3:
                            continue

                        hdu = fits.open(cal_file)
                        if band_type == 'nircam':
                            if hdu[0].header['FILTER'].strip() == jwst_filter and \
                                    hdu[0].header['PUPIL'].strip() == pupil:
                                os.system('cp %s %s/' % (cal_file, input_dir))
                        elif band_type == 'miri':
                            if hdu[0].header['FILTER'].strip() == jwst_filter:
                                os.system('cp %s %s/' % (cal_file, input_dir))
                        hdu.close()

                # Run lv3 asn generation
                asn_file = self.run_asn3(directory=input_dir,
                                         band=band)

                # Run pipeline
                self.run_pipeline(band=band,
                                  input_dir=input_dir,
                                  output_dir=output_dir,
                                  asn_file=asn_file,
                                  pipeline_stage='lv3')

                if self.process_bgr_like_science:
                    asn_file_bgr = self.run_asn3(directory=input_dir,
                                                 band=band, process_bgr_like_science=True)
                    output_dir_bgr = os.path.join(self.reprocess_dir, self.galaxy, band, 'bgr')
                    self.run_pipeline(band=band,
                                      input_dir=input_dir,
                                      output_dir=output_dir_bgr,
                                      asn_file=asn_file_bgr,
                                      pipeline_stage='lv3')

            if self.do_astrometric_alignment:
                self.logger.info('-> Astrometric alignment')

                if self.use_field_in_lev3 is None:

                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'lv3')

                else:
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy, band,
                                             'lv3_field_' + '_'.join(np.atleast_1d(
                                                 self.use_field_in_lev3).astype(str)))

                if band in self.alignment_mapping.keys():
                    self.align_wcs_to_jwst(input_dir,
                                           band)
                else:
                    self.align_wcs_to_ref(input_dir,
                                          )

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

        for directory in [plot_dir, pca_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        with mp.get_context('fork').Pool(self.procs) as pool:

            results = []

            for result in tqdm(pool.imap(partial(parallel_destripe,
                                                 band=band,
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

        for hdu_name in in_files:
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

        for hdu_name in in_files:
            hdu = fits.open(hdu_name)

            out_name = os.path.join(out_dir,
                                    os.path.split(hdu_name)[-1])

            if os.path.exists(out_name):
                return True

            hdu['SCI'].data[735:, :290] = 0
            hdu['DQ'].data[735:, :290] = 1  # Masks the coronagraph area
            hdu.writeto(out_name, overwrite=True)

            hdu.close()

    def wcs_adjust(self,
                   input_dir,
                   output_dir,
                   band):
        """Adjust WCS so the tweakreg solution is closer to 0. Should be run on cal.fits files

        Args:
            input_dir (str): Input directory
            output_dir (str): Where to save files to
            band (str): JWST filter
        """

        input_files = glob.glob(os.path.join(input_dir,
                                             '*cal.fits')
                                )
        input_files.sort()

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
            group = os.path.split(input_file)[-1]
            group = '_'.join(group.split('_')[:3])

            try:
                wcs_adjust_vals = self.wcs_adjust_dict[band][group]
                matrix = wcs_adjust_vals['matrix']
                shift = wcs_adjust_vals['shift']
            except KeyError:
                self.logger.info('No WCS adjust info found for %s. Defaulting to no shift' % group)
                matrix = [[1, 0], [0, 1]]
                shift = [0, 0]

            im.set_correction(matrix=matrix, shift=shift)

            input_im.meta.wcs = im.wcs

            try:
                update_fits_wcsinfo(
                    input_im,
                    max_pix_error=0.05
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
                 process_bgr_like_science=False
                 ):
        """Setup asn lv2 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
            * process_bgr_like_science (bool): if True, than additionally process the offset images
                in the same way as the science (for testing purposes only)
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

        if not os.path.exists(asn_lv2_filename) or self.overwrite_lv2:

            tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', "Program"],
                        dtype=[str, str, str, str, str, float, str, str])

            all_fits_files = glob.glob('*%s_rate.fits' % band_ext)
            for f in all_fits_files:
                tab.add_row(parse_fits_to_table(f,
                                                check_bgr=check_bgr,
                                                check_type=self.bgr_check_type)
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
            if (band_type == 'miri') and process_bgr_like_science:
                for row_id, row in enumerate(bgr_tab):
                    json_content['products'].append({
                        'name': f'offset_{band}_{row_id + 1}',
                        'members': [
                            {'expname': row['File'],
                             'exptype': 'science',
                             'exposerr': 'null'}
                        ]
                    })

            # Associate background files, but only for MIRI
            if band_type == 'miri':
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
                 process_bgr_like_science=False):
        """Setup asn lv3 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
            * process_bgr_like_science (bool): Additionally process the offset images in the same way as the science
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

        if not os.path.exists(asn_lv3_filename) or self.overwrite_lv3:

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
                                                check_type=self.bgr_check_type)
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
                            "products": [{'name': '%s_%s_lv3_%s' % (self.galaxy.lower(), band_type, band.lower()),
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
                     ):
        """Run JWST pipeline.

        Args:
            * band (str): JWST filter
            * input_dir (str): Files associated to asn_file
            * output_dir (str): Where to save the pipeline outputs
            * asn_file (str): Path to asn file. For lv1, set this to None
            * pipeline_stage (str): Pipeline processing stage. Should be 'lv1', 'lv2', or 'lv3'
        """

        if band in NIRCAM_BANDS:
            band_type = 'nircam'
        elif band in MIRI_BANDS:
            band_type = 'miri'
        else:
            raise Warning('Band %s not recognised!' % band)

        orig_dir = os.getcwd()

        os.chdir(input_dir)

        if pipeline_stage == 'lv1':

            if self.overwrite_lv1:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or self.overwrite_lv1:

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

                    detector1 = attribute_setter(detector1, self.lv1_parameter_dict, band)    
                    # for key in self.lv1_parameter_dict.keys():

                    #     value = parse_parameter_dict(self.lv1_parameter_dict,
                    #                                  key,
                    #                                  band)
                    #     if value == 'VAL_NOT_FOUND':
                    #         continue

                    #     recursive_setattr(detector1, key, value)

                    # Run the level 1 pipeline
                    detector1.run(uncal_file)

        elif pipeline_stage == 'lv2':

            if self.overwrite_lv2:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or self.overwrite_lv2:

                os.system('rm -rf %s' % output_dir)

                config = calwebb_image2.Image2Pipeline.get_config_from_reference(asn_file)
                im2 = calwebb_image2.Image2Pipeline.from_config_section(config)
                im2.output_dir = output_dir
                if not os.path.isdir(im2.output_dir):
                    os.makedirs(im2.output_dir)


                im2 = attribute_setter(im2, self.lv2_parameter_dict, band)

                # for key in self.lv2_parameter_dict.keys():

                #     value = parse_parameter_dict(self.lv2_parameter_dict,
                #                                  key,
                #                                  band)
                #     if value == 'VAL_NOT_FOUND':
                #         continue

                #     recursive_setattr(im2, key, value)

                if self.updated_flats_dir is not None:
                    my_flat = [f for f in glob.glob(os.path.join(self.updated_flats_dir, "*.fits")) if band in f]
                    if len(my_flat) != 0:
                        im2.flat_field.user_supplied_flat = my_flat[0]
                # Run the level 2 pipeline
                im2.run(asn_file)

        elif pipeline_stage == 'lv3':

            if self.overwrite_lv3:
                os.system('rm -rf %s' % output_dir)

            output_fits = '%s_%s_lv3_%s_i2d.fits' % (self.galaxy.lower(), band_type, band.lower())
            output_file = os.path.join(output_dir, output_fits)

            if not os.path.exists(output_file) or self.overwrite_lv3:

                os.system('rm -rf %s' % output_dir)

                if not os.path.isdir(output_dir):
                    os.makedirs(output_dir)

                # FWHM should be set per-band for both tweakreg and source catalogue
                fwhm_pix = FWHM_PIX[band]

                # If we're degrouping short NIRCAM observations, we still want to align grouped
                if self.degroup_short_nircam:
                    tweakreg = TweakRegStep()
                    tweakreg.output_dir = output_dir
                    tweakreg.save_results = False
                    tweakreg.kernel_fwhm = fwhm_pix * 2

                    try:
                        tweakreg_params = self.lv3_parameter_dict['tweakreg']
                    except KeyError:
                        pass
                    
                    for tweakreg_key in tweakreg_params:
                        value = parse_parameter_dict(tweakreg_params,
                                                     tweakreg_key,
                                                     band)
                                            
                        if value == 'VAL_NOT_FOUND':
                            continue
                        
                        recursive_setattr(tweakreg, tweakreg_key, value)
                    asn_file = tweakreg.run(asn_file)

                # Now run through the rest of the pipeline

                config = calwebb_image3.Image3Pipeline.get_config_from_reference(asn_file)
                im3 = calwebb_image3.Image3Pipeline.from_config_section(config)
                im3.output_dir = output_dir

                im3.tweakreg.kernel_fwhm = fwhm_pix * 2
                im3.source_catalog.kernel_fwhm = fwhm_pix * 2

                im3 = attribute_setter(im3, self.lv3_parameter_dict, band)
                
                if self.degroup_short_nircam:

                    # Make sure we skip tweakreg since we've already done it
                    im3.tweakreg.skip = True

                    # Degroup the short NIRCAM observations, to avoid background issues
                    if int(band[1:4]) <= 212 and band_type == 'nircam':
                        degroup = True
                    else:
                        degroup = False

                    if degroup:
                        for i, model in enumerate(asn_file._models):
                            model.meta.observation.exposure_number = str(i)
                # Run the level 3 pipeline
                im3.run(asn_file)

        else:

            raise Warning('Pipeline stage %s not recognised!' % pipeline_stage)

        os.chdir(orig_dir)

    def align_wcs_to_ref(self,
                         input_dir,
                         ):
        """Align JWST image to external references. Either a table or an image

        Args:
            * input_dir (str): Directory to find files to align
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

            if not os.path.exists(source_cat_name) or self.overwrite_astrometric_ref_cat:

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

            if not os.path.exists(aligned_file) or self.overwrite_astrometric_alignment:
                jwst_hdu = fits.open(jwst_file)

                jwst_data = copy.deepcopy(jwst_hdu['SCI'].data)
                jwst_data[jwst_data == 0] = np.nan

                # Read in the source catalogue from the pipeline

                source_cat_name = jwst_file.replace('i2d.fits', 'cat.ecsv')
                sources = Table.read(source_cat_name, format='ascii.ecsv')
                # convenience for CARTA viewing.
                sources.write(source_cat_name.replace('.ecsv','.fits'), overwrite=True)

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
                jwst_tab['x'] = sources['xcentroid']
                jwst_tab['y'] = sources['ycentroid']
                jwst_tab['ra'] = sources['sky_centroid'].ra.value
                jwst_tab['dec'] = sources['sky_centroid'].dec.value

                # Run a match
                match = XYXYMatch(
                    searchrad=self.tpmatch_searchrad,
                    separation=self.tpmatch_separation,
                    tolerance=self.tpmatch_tolerance,
                    use2dhist=self.tpmatch_use2dhist,
                )
                ref_idx, jwst_idx = match(ref_tab, jwst_tab, wcs_jwst_corrector)

                # Do alignment
                wcs_aligned_fit = fit_wcs(ref_tab[ref_idx],
                                          jwst_tab[jwst_idx],
                                          wcs_jwst_corrector,
                                          fitgeom=self.tpmatch_fitgeom,
                                          nclip=self.tpmatch_nclip,
                                          sigma=(self.tpmatch_sigma, 'rmse'),
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

                fit_info = wcs_aligned_fit.meta['fit_info']
                fit_mask = fit_info['fitmask']

                # Pull out useful alignment info to the table -- HST x/y/RA/Dec, JWST x/y/RA/Dec (corrected and
                # uncorrected)
                aligned_tab = Table()

                # Catch if there's only RA/Dec in the reference table
                if 'xcentroid' in ref_tab.colnames:
                    aligned_tab['xcentroid_ref'] = ref_tab[ref_idx]['xcentroid']
                    aligned_tab['ycentroid_ref'] = ref_tab[ref_idx]['ycentroid']
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
                          band):
        """Internally align image to already aligned JWST one via cross-correlation

        Args:
            * input_dir (str): Directory to find files to align
            * band (str): JWST band to align
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
                                    self.galaxy,
                                    ref_band,
                                    ref_dir,
                                    '%s_%s_lv3_%s_i2d_align.fits' % (self.galaxy, ref_band_type, ref_band.lower()))

        if not os.path.exists(ref_hdu_name):
            raise Warning('reference HDU to align not found!')

        for jwst_file in jwst_files:

            aligned_file = jwst_file.replace('.fits', '_align.fits')

            if not os.path.exists(aligned_file) or self.overwrite_astrometric_alignment:
                ref_hdu = fits.open(ref_hdu_name)
                jwst_hdu = fits.open(jwst_file)

                wcs_jwst = HSTWCS(jwst_hdu, 'SCI')
                wcs_jwst_corrector = FITSWCS(wcs_jwst)

                ref_data = copy.deepcopy(ref_hdu['SCI'].data)
                jwst_data = copy.deepcopy(jwst_hdu['SCI'].data)

                ref_err = copy.deepcopy(ref_hdu['ERR'].data)
                jwst_err = copy.deepcopy(jwst_hdu['ERR'].data)

                ref_data[ref_data == 0] = np.nan
                jwst_data[jwst_data == 0] = np.nan

                # Reproject the ref HDU to the image to align

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

                jwst_hdu.writeto(aligned_file, overwrite=True)

                jwst_hdu.close()

                ref_hdu.close()
