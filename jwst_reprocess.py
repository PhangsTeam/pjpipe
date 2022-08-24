import copy
import functools
import glob
import json
import logging
import os
import shutil
import time
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count, set_start_method

set_start_method('fork')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, QTable
from drizzlepac import updatehdr
from image_registration import cross_correlation_shifts
from jwst import datamodels
from photutils import make_source_mask
from photutils.detection import DAOStarFinder
from reproject import reproject_interp
from stwcs.wcsutil import HSTWCS
from tqdm import tqdm
from tweakwcs import fit_wcs, TPMatch, FITSWCS

from nircam_destriping import NircamDestriper

jwst = None
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
                      quadrants=True,
                      destriping_method='row_median',
                      sigma=3,
                      npixels=3,
                      max_iters=20,
                      dilate_size=11,
                      median_filter_scales=None,
                      pca_components=50,
                      pca_reconstruct_components=10,
                      pca_diffuse=False,
                      pca_dir=None,
                      just_sci_hdu=False,
                      out_dir=None,
                      plot_dir=None,
                      ):
    """Function to parallelise destriping"""

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
                                  quadrants=quadrants,
                                  destriping_method=destriping_method,
                                  sigma=sigma,
                                  npixels=npixels,
                                  max_iters=max_iters,
                                  dilate_size=dilate_size,
                                  median_filter_scales=median_filter_scales,
                                  pca_components=pca_components,
                                  pca_reconstruct_components=pca_reconstruct_components,
                                  pca_diffuse=pca_diffuse,
                                  pca_file=pca_file,
                                  just_sci_hdu=just_sci_hdu,
                                  plot_dir=plot_dir,
                                  )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
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
            raise NotImplementedError('Not yet implemented!')
            # f_type = 'bgr'
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
    """Pull values out of a parameter dictionary"""

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
                 degroup_short_nircam=True,
                 bgr_check_type='parallel_off',
                 astrometric_alignment_type='image',
                 astrometric_alignment_image=None,
                 astrometric_alignment_table=None,
                 alignment_mapping=None,
                 do_all=True,
                 do_lv1=False,
                 do_lv2=False,
                 do_destriping=False,
                 do_lyot_adjust=None,
                 do_lv3=False,
                 do_astrometric_alignment=False,
                 overwrite_all=False,
                 overwrite_lv1=False,
                 overwrite_lv2=False,
                 overwrite_destriping=False,
                 overwrite_lyot_adjust=False,
                 overwrite_lv3=False,
                 overwrite_astrometric_alignment=False,
                 overwrite_astrometric_ref_cat=False,
                 correct_lv1_wcs=False,
                 crds_url='https://jwst-crds.stsci.edu',
                 procs=None
                 ):
        """JWST reprocessing routines.

        Will run through whole JWST pipeline, allowing for fine-tuning along the way.

        It's worth talking a little about how parameter dictionaries are passed. They should be of the form

                {'parameter': value}

        where parameter is how the pipeline names it, e.g. 'save_results', 'tweakreg.fitgeometry'. Because you might
        want to break these out per observing mode, you can also pass a dict, like

                {'parameter': {'miri': miri_val, 'nircam': nircam_val}}

        where the acceptable variants are 'miri', 'nircam', 'nircam_long', and 'nircam_short'. As many bits of the
        pipeline require a number in arcsec rather than pixels, you can pass a value as 'Xpix', and it will parse
        according to the band you're processing.

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
            * degroup_short_nircam (bool): Will degroup short wavelength NIRCAM observations for all steps beyond
                relative alignment. This can alleviate steps between mosaic pointings
            * bgr_check_type (str): Method to check if MIRI obs is science or background. Options are 'parallel_off' and
                off_in_name. Defaults to 'parallel_off'
            * astrometric_alignment_image (str): Path to image to align astrometry to
            * astrometric_alignment_table (str): Path to table to align astrometry to
            * alignment_mapping (dict): Dictionary to map basing alignments off cross-correlation with other aligned
                band. Should be of the form {'band': 'reference_band'}
            * do_all (bool): Do all processing steps. Defaults to True
            * do_lv1 (bool): Run lv1 pipeline. Defaults to False
            * do_lv2 (bool): Run lv2 pipeline. Defaults to False
            * do_destriping (bool): Run destriping algorithm on lv2 data. Defaults to False
            * do_lyot_adjust (str): How to deal with the MIRI coronagraph. Options are 'mask', which masks it out, or
                'adjust', which will adjust the background level to match the main array
            * do_lv3 (bool): Run lv3 pipeline. Defaults to False
            * do_astrometric_alignment (bool): Run astrometric alignment on lv3 data. Defaults to False
            * overwrite_all (bool): Whether to overwrite everything. Defaults to False
            * overwrite_lv1 (bool): Whether to overwrite lv1 data. Defaults to False
            * overwrite_lv2 (bool): Whether to overwrite lv2 data. Defaults to False
            * overwrite_destriping (bool): Whether to overwrite destriped data. Defaults to False
            * overwrite_lyot_adjust (bool): Whether to overwrite MIRI coronagraph edits. Defaults to False
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

        TODO:
            * Update destriping algorithm as we improve it
            * Record alignment parameters into the fits header

        """

        os.environ['CRDS_SERVER_URL'] = crds_url
        os.environ['CRDS_PATH'] = crds_dir

        global jwst, calwebb_detector1, calwebb_image2, calwebb_image3, TweakRegStep
        import jwst
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

                'skymatch.skip': {'miri': True},
                'skymatch.skymethod': 'global+match',
                'skymatch.subtract': True,
                'skymatch.skystat': 'median',
                'skymatch.nclip': 20,
                'skymatch.lsigma': 3,
                'skymatch.usigma': 3,

                'outlier_detection.in_memory': True,

                'resample.rotation': 0.0,
                'resample.in_memory': True,

                'source_catalog.snr_threshold': 3,
                'source_catalog.npixels': 5,
                'source_catalog.bkg_boxsize': 100,
                'source_catalog.deblend': True
            }

        self.lv3_parameter_dict = lv3_parameter_dict

        self.degroup_short_nircam = degroup_short_nircam

        if do_all:
            do_lv1 = True
            do_lv2 = True
            do_destriping = True
            do_lyot_adjust = 'adjust'
            do_lv3 = True
            do_astrometric_alignment = True

        self.do_lv1 = do_lv1
        self.do_lv2 = do_lv2
        self.do_destriping = do_destriping
        self.do_lyot_adjust = do_lyot_adjust
        self.do_lv3 = do_lv3
        self.do_astrometric_alignment = do_astrometric_alignment
        self.alignment_mapping = alignment_mapping

        self.astrometric_alignment_type = astrometric_alignment_type
        self.astrometric_alignment_image = astrometric_alignment_image
        self.astrometric_alignment_table = astrometric_alignment_table

        self.bgr_check_type = bgr_check_type

        if overwrite_all:
            overwrite_lv1 = True
            overwrite_lv2 = True
            overwrite_destriping = True
            overwrite_lyot_adjust = True
            overwrite_lv3 = True
            overwrite_astrometric_alignment = True

        self.overwrite_all = overwrite_all
        self.overwrite_lv1 = overwrite_lv1
        self.overwrite_lv2 = overwrite_lv2
        self.overwrite_destriping = overwrite_destriping
        self.overwrite_lyot_adjust = overwrite_lyot_adjust
        self.overwrite_lv3 = overwrite_lv3
        self.overwrite_astrometric_alignment = overwrite_astrometric_alignment
        self.overwrite_astrometric_ref_cat = overwrite_astrometric_ref_cat

        self.correct_lv1_wcs = correct_lv1_wcs

        if procs is None:
            procs = cpu_count() // 2

        self.procs = procs

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

    def run_all(self):
        """Run the whole pipeline reprocess"""

        self.logger.info('Reprocessing %s' % self.galaxy)

        for band in self.bands:

            if band in NIRCAM_BANDS:
                band_type = 'nircam'
                do_destriping = self.do_destriping
                do_lyot_adjust = None
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

                            hdu = fits.open(uncal_file)
                            if hdu[0].header['FILTER'].strip() == band:
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
                            if hdu[0].header['FILTER'].strip() == band:
                                os.system('cp %s %s/' % (rate_file, rate_dir))
                            hdu.close()

                # Run lv2 asn generation
                asn_file = self.run_asn2(directory=rate_dir,
                                         band=band)

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
                        if hdu[0].header['FILTER'].strip() == band:
                            cal_files.append(cal_file)
                        hdu.close()

                cal_files.sort()
                self.run_destripe(files=cal_files,
                                  out_dir=destripe_dir
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
                        if hdu[0].header['FILTER'].strip() == band:
                            cal_files.append(cal_file)
                        hdu.close()

                cal_files.sort()

                if do_lyot_adjust == 'adjust':
                    self.adjust_lyot(in_files=cal_files,
                                     out_dir=lyot_adjust_dir)
                elif do_lyot_adjust == 'mask':
                    self.mask_lyot(in_files=cal_files,
                                   out_dir=lyot_adjust_dir)

            if self.do_lv3:

                self.logger.info('-> Level 3')

                output_dir = os.path.join(self.reprocess_dir,
                                          self.galaxy,
                                          band,
                                          'lv3')

                if self.overwrite_lv3:
                    os.system('rm -rf %s' % output_dir)

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

                    cal_files = glob.glob(os.path.join(self.raw_dir,
                                                       self.galaxy,
                                                       'mastDownload',
                                                       'JWST',
                                                       '*%s' % band_ext,
                                                       '*%s_cal.fits' % band_ext)
                                          )

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
                        if hdu[0].header['FILTER'].strip() == band:
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

            if self.do_astrometric_alignment:
                self.logger.info('-> Astrometric alignment')

                input_dir = os.path.join(self.reprocess_dir,
                                         self.galaxy,
                                         band,
                                         'lv3')

                if band in self.alignment_mapping.keys():
                    self.align_wcs_to_jwst(input_dir,
                                           band)
                else:
                    self.align_wcs_to_ref(input_dir,
                                          )

    def run_destripe(self,
                     files,
                     out_dir,
                     ):
        """Run destriping algorithm, looping over calibrated files

        Args:
            * files (list): List of files to loop over
            * out_dir (str): Where to save destriped files to
        """

        plot_dir = os.path.join(out_dir, 'plots')
        pca_dir = os.path.join(out_dir, 'pca')

        for directory in [plot_dir, pca_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

        with Pool(self.procs) as pool:

            results = []

            for result in tqdm(pool.imap_unordered(partial(parallel_destripe,
                                                           quadrants=True,
                                                           destriping_method='pca+median',
                                                           dilate_size=7,
                                                           pca_reconstruct_components=5,
                                                           pca_diffuse=True,
                                                           pca_dir=pca_dir,
                                                           out_dir=out_dir,
                                                           plot_dir=plot_dir,
                                                           ),
                                                   files),
                               total=len(files), ascii=True):
                results.append(result)

            # median_filter_scales = [7, 31, 63, 127, 511]
            #
            # for result in tqdm(pool.imap_unordered(partial(parallel_destripe,
            #                                                quadrants=False,
            #                                                destriping_method='median_filter',
            #                                                dilate_size=7,
            #                                                out_dir=out_dir,
            #                                                ),
            #                                        files),
            #                    total=len(files), ascii=True):
            #     results.append(result)

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
            lyot = copy.deepcopy(hdu['SCI'].data[750:, :280])
            lyot_dq = copy.deepcopy(hdu['DQ'].data[750:, :280])
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

            hdu['SCI'].data[750:, :280] += (bgr_image - bgr_lyot)
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

            hdu['DQ'].data[750:, :280] = 513  # Masks the coronagraph area
            hdu.writeto(out_name, overwrite=True)

            hdu.close()

    def run_asn2(self,
                 directory=None,
                 band=None,
                 ):
        """Setup asn lv2 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
        """

        if directory is None:
            raise Warning('Directory should be specified!')
        if band is None:
            raise Warning('Band should be specified!')

        if band in NIRCAM_BANDS:
            band_type = 'nircam'
            check_bgr = False
        elif band in MIRI_BANDS:
            band_type = 'miri'
            check_bgr = True
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

            # Associate background files
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
                 band=None):
        """Setup asn lv3 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
        """

        if band is None:
            raise Warning('Band must be specified!')

        if band in NIRCAM_BANDS:
            band_type = 'nircam'
            check_bgr = False
        elif band in MIRI_BANDS:
            band_type = 'miri'
            check_bgr = True
        else:
            raise Warning('Band %s not recognised!' % band)

        band_ext = BAND_EXTS[band_type]

        orig_dir = os.getcwd()

        os.chdir(directory)

        asn_lv3_filename = 'asn_lv3_%s.json' % band

        if not os.path.exists(asn_lv3_filename) or self.overwrite_lv3:

            lv2_files = glob.glob('*%s_cal.fits' % band_ext)
            tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', 'Program'],
                        dtype=[str, str, str, str, str, float, str, str])

            for f in lv2_files:
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
            sci_tab = tab[tab['Type'] == 'sci']

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

                    # There appears to be a bug that sometimes WCS info isn't populated through to the uncal files.
                    # Fix that here
                    if self.correct_lv1_wcs:
                        if 'MAST_API_TOKEN' not in os.environ.keys():
                            os.environ['MAST_API_TOKEN'] = input('Input MAST API token: ')

                        uncal_hdu = fits.open(uncal_file)

                        qual = uncal_hdu[0].header['ENGQLPTG']

                        if qual == 'PLANNED':
                            os.system('set_telescope_pointing.py %s' % uncal_file)

                    detector1 = calwebb_detector1.Detector1Pipeline()

                    # Set some parameters that pertain to the
                    # entire pipeline
                    detector1.output_dir = output_dir
                    if not os.path.isdir(detector1.output_dir):
                        os.makedirs(detector1.output_dir)

                    for key in self.lv1_parameter_dict.keys():

                        value = parse_parameter_dict(self.lv1_parameter_dict,
                                                     key,
                                                     band)
                        if value == 'VAL_NOT_FOUND':
                            continue

                        recursive_setattr(detector1, key, value)

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

                    # Run the level 1 pipeline
                    detector1.run(uncal_file)

        elif pipeline_stage == 'lv2':

            if self.overwrite_lv2:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or self.overwrite_lv2:

                os.system('rm -rf %s' % output_dir)

                im2 = calwebb_image2.Image2Pipeline()
                im2.output_dir = output_dir
                if not os.path.isdir(im2.output_dir):
                    os.makedirs(im2.output_dir)

                for key in self.lv2_parameter_dict.keys():

                    value = parse_parameter_dict(self.lv2_parameter_dict,
                                                 key,
                                                 band)
                    if value == 'VAL_NOT_FOUND':
                        continue

                    recursive_setattr(im2, key, value)

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
                    tweakreg.kernel_fwhm = fwhm_pix

                    for key in self.lv3_parameter_dict.keys():
                        if key.split('.')[0] == 'tweakreg':

                            tweakreg_key = '.'.join(key.split('.')[1:])

                            value = parse_parameter_dict(self.lv3_parameter_dict,
                                                         key,
                                                         band)
                            if value == 'VAL_NOT_FOUND':
                                continue

                            recursive_setattr(tweakreg, tweakreg_key, value)

                    asn_file = tweakreg.run(asn_file)

                # Now run through the rest of the pipeline

                im3 = calwebb_image3.Image3Pipeline()
                im3.output_dir = output_dir

                im3.tweakreg.kernel_fwhm = fwhm_pix
                im3.source_catalog.kernel_fwhm = fwhm_pix

                for key in self.lv3_parameter_dict.keys():

                    value = parse_parameter_dict(self.lv3_parameter_dict,
                                                 key,
                                                 band)
                    if value == 'VAL_NOT_FOUND':
                        continue

                    recursive_setattr(im3, key, value)

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

            # if 'parallax' in astro_table.colnames:
            #     # This should be a GAIA query, so cut down based on whether there is a parallax measurement
            #     idx = np.where(~np.isnan(astro_table['parallax']))
            #     # idx = np.where(np.logical_and(astro_table['ra_error'].value < 1,
            #     #                               astro_table['dec_error'].value < 1))
            #     astro_table = astro_table[idx]

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

                # Filter out extended
                sources = sources[~sources['is_extended']]

                # Convert sources into a reference catalogue
                wcs_jwst = HSTWCS(jwst_hdu, 'SCI')
                wcs_jwst_corrector = FITSWCS(wcs_jwst)

                jwst_tab = Table()
                jwst_tab['x'] = sources['xcentroid']
                jwst_tab['y'] = sources['ycentroid']
                jwst_tab['ra'] = sources['sky_centroid'].ra.value
                jwst_tab['dec'] = sources['sky_centroid'].dec.value

                # Run a match with fairly strict tolerance
                match = TPMatch(
                    searchrad=10,
                    separation=0.000001,
                    tolerance=0.7,
                    use2dhist=True,
                )
                ref_idx, jwst_idx = match(ref_tab, jwst_tab, wcs_jwst_corrector)

                # Do alignment

                wcs_aligned_fit = fit_wcs(ref_tab[ref_idx],
                                          jwst_tab[jwst_idx],
                                          wcs_jwst_corrector,
                                          fitgeom='shift',
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

                # Pull out useful alignment info to the table -- HST x/y/RA/Dec, JWST x/y/RA/Dec (corrected and
                # uncorrected)
                aligned_tab = Table()

                # Catch if there's only RA/Dec in the reference table
                if 'xcentroid' in ref_tab.colnames:
                    aligned_tab['xcentroid_ref'] = ref_tab[ref_idx]['xcentroid']
                    aligned_tab['ycentroid_ref'] = ref_tab[ref_idx]['ycentroid']
                aligned_tab['ra_ref'] = ref_tab[ref_idx]['RA']
                aligned_tab['dec_ref'] = ref_tab[ref_idx]['DEC']

                # Since we're pulling from the source catalogue, these should all exist
                aligned_tab['xcentroid_jwst'] = jwst_tab[jwst_idx]['x']
                aligned_tab['ycentroid_jwst'] = jwst_tab[jwst_idx]['y']
                aligned_tab['ra_jwst_uncorr'] = jwst_tab[jwst_idx]['ra']
                aligned_tab['dec_jwst_uncorr'] = jwst_tab[jwst_idx]['dec']

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

        ref_hdu_name = os.path.join(self.reprocess_dir,
                                    self.galaxy,
                                    ref_band,
                                    'lv3',
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
