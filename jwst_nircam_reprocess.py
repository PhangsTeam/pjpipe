import copy
import glob
import json
import os
import time
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table, QTable
from drizzlepac import updatehdr
from photutils.detection import DAOStarFinder
from stwcs.wcsutil import HSTWCS
from tweakwcs import fit_wcs, TPMatch, FITSWCS

from nircam_destriping import NircamDestriper

jwst = None
calwebb_detector1 = None
calwebb_image2 = None
calwebb_image3 = None


def parse_fits_to_table(file):
    """Pull necessary info out of fits headers"""
    f_type = 'sci'
    with fits.open(file) as hdu:
        return file, f_type, hdu[0].header['OBSERVTN'], hdu[0].header['filter'], \
               hdu[0].header['DATE-BEG'], hdu[0].header['DURATION'], \
               hdu[0].header['OBSLABEL'].lower().strip(), hdu[0].header['PROGRAM']


class NircamReprocess:

    def __init__(self,
                 crds_path,
                 galaxy,
                 raw_dir,
                 reprocess_dir,
                 bands=None,
                 astrometric_alignment_image=None,
                 do_all=False,
                 do_lv1=False,
                 do_lv2=False,
                 do_destriping=False,
                 do_lv3=False,
                 do_astrometric_alignment=False,
                 overwrite_all=False,
                 overwrite_lv1=False,
                 overwrite_lv2=False,
                 overwrite_destriping=False,
                 overwrite_lv3=False,
                 overwrite_astrometric_alignment=False,
                 crds_url='https://jwst-crds-pub.stsci.edu'
                 ):
        """NIRCAM reprocessing routines.

        Will run through whole NIRCAM pipeline, allowing for fine-tuning along the way

        Args:
            * crds_path (str): Path to CRDS data
            * galaxy (str): Galaxy to run reprocessing for
            * raw_dir (str): Path to raw data
            * reprocess_dir (str): Path to reprocess data into
            * bands (list): JWST filters to loop over
            * astrometric_alignment_image (str): Path to image to align astrometry to
            * do_all (bool): Do all processing steps
            * do_lv1 (bool): Run lv1 pipeline
            * do_lv2 (bool): Run lv2 pipeline
            * do_destriping (bool): Run destriping algorithm on lv2 data
            * do_lv3 (bool): Run lv3 pipeline
            * do_astrometric_alignment (bool): Run astrometric alignment on lv3 data
            * overwrite (bool): Whether to overwrite destriping/reprocessing. Defaults to False
            * crds_url (str): URL to get CRDS files from. Defaults to 'https://jwst-crds-pub.stsci.edu'

        TODO:
            * Update destriping algorithm as we improve it
            * Print out some useful info as stuff runs

        """

        os.environ['CRDS_SERVER_URL'] = crds_url
        os.environ['CRDS_PATH'] = crds_path

        global jwst, calwebb_detector1, calwebb_image2, calwebb_image3
        import jwst
        from jwst.pipeline import calwebb_detector1, calwebb_image2, calwebb_image3

        self.galaxy = galaxy

        if bands is None:
            bands = ['F200W', 'F300M', 'F335M', 'F360M']

        self.bands = bands

        self.astrometric_alignment_image = astrometric_alignment_image

        self.raw_dir = raw_dir
        self.reprocess_dir = reprocess_dir

        if do_all:
            do_lv1 = True
            do_lv2 = True
            do_destriping = True
            do_lv3 = True
            do_astrometric_alignment = True

        self.do_lv1 = do_lv1
        self.do_lv2 = do_lv2
        self.do_destriping = do_destriping
        self.do_lv3 = do_lv3
        self.do_astrometric_alignment = do_astrometric_alignment

        if overwrite_all:
            overwrite_lv1 = True
            overwrite_lv2 = True
            overwrite_destriping = True
            overwrite_lv3 = True
            overwrite_astrometric_alignment = True

        self.overwrite_all = overwrite_all
        self.overwrite_lv1 = overwrite_lv1
        self.overwrite_lv2 = overwrite_lv2
        self.overwrite_destriping = overwrite_destriping
        self.overwrite_lv3 = overwrite_lv3
        self.overwrite_astrometric_alignment = overwrite_astrometric_alignment

    def run_all(self):
        """Run the whole pipeline reprocess"""

        for band in self.bands:

            if self.overwrite_all:
                os.system('rm -rf %s' % os.path.join(self.reprocess_dir,
                                                     self.galaxy,
                                                     band))

            if self.do_lv1:
                uncal_dir = os.path.join(self.reprocess_dir,
                                         self.galaxy,
                                         band,
                                         'uncal'
                                         )

                if not os.path.exists(uncal_dir):
                    os.makedirs(uncal_dir)

                rate_dir = os.path.join(self.reprocess_dir,
                                        self.galaxy,
                                        band,
                                        'rate')

                uncal_files = glob.glob(os.path.join(self.raw_dir,
                                                     self.galaxy,
                                                     'mastDownload',
                                                     'JWST',
                                                     '*nrc*',
                                                     '*nrc*_uncal.fits')
                                        )
                uncal_files.sort()

                for uncal_file in uncal_files:

                    uncal_fits_name = uncal_file.split(os.path.sep)[-1]
                    hdu_out_name = os.path.join(uncal_dir, uncal_fits_name)

                    if not os.path.exists(hdu_out_name) or self.overwrite_lv1:

                        hdu = fits.open(uncal_file)
                        if hdu[0].header['FILTER'].strip() == band:
                            hdu.writeto(hdu_out_name, overwrite=True)

                if self.overwrite_lv1:
                    os.system('rm -rf %s' % rate_dir)

                # Run pipeline
                self.run_pipeline(band=band,
                                  input_dir=uncal_dir,
                                  output_dir=rate_dir,
                                  asn_file=None,
                                  pipeline_stage='lev1')

            if self.do_lv2:

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
                                                        '*nrc*',
                                                        '*nrc*_rate.fits')
                                           )

                    for rate_file in rate_files:

                        fits_name = rate_file.split(os.path.sep)[-1]
                        if not os.path.exists(os.path.join(rate_dir, fits_name)) or self.overwrite_lv2:

                            hdu = fits.open(rate_file)[0]
                            if hdu.header['FILTER'].strip() == band:
                                os.system('cp %s %s/' % (rate_file, rate_dir))

                # Run lv2 asn generation
                asn_file = self.run_asn2(directory=rate_dir,
                                         band=band)

                # Run pipeline
                self.run_pipeline(band=band,
                                  input_dir=rate_dir,
                                  output_dir=cal_dir,
                                  asn_file=asn_file,
                                  pipeline_stage='lev2')

            if self.do_destriping:

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
                                                       '*nrc*_cal.fits')
                                          )

                else:
                    initial_cal_files = glob.glob(os.path.join(self.raw_dir,
                                                               self.galaxy,
                                                               'mastDownload',
                                                               'JWST',
                                                               '*nrc*',
                                                               '*nrc*_cal.fits')
                                                  )

                    cal_files = []

                    for cal_file in initial_cal_files:

                        hdu = fits.open(cal_file)[0]
                        if hdu.header['FILTER'].strip() == band:
                            cal_files.append(cal_file)

                cal_files.sort()
                self.run_destripe(files=cal_files,
                                  output_dir=destripe_dir
                                  )

            if self.do_lv3:

                output_dir = os.path.join(self.reprocess_dir,
                                          self.galaxy,
                                          band,
                                          'lev3')

                if self.overwrite_lv3:
                    os.system('rm -rf %s' % output_dir)

                if self.do_lv2 and not self.do_destriping:
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'cal')
                elif self.do_destriping:
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'destripe')
                else:
                    # At this point, we pull the cal files from the raw directories
                    input_dir = os.path.join(self.reprocess_dir,
                                             self.galaxy,
                                             band,
                                             'lev2')

                    if not os.path.exists(input_dir):
                        os.makedirs(input_dir)

                    cal_files = glob.glob(os.path.join(self.raw_dir,
                                                       self.galaxy,
                                                       'mastDownload',
                                                       'JWST',
                                                       '*nrc*',
                                                       '*nrc*_cal.fits')
                                          )

                    for cal_file in cal_files:

                        fits_name = cal_file.split(os.path.sep)[-1]
                        if os.path.exists(os.path.join(input_dir, fits_name)) or self.overwrite_lv3:
                            continue

                        hdu = fits.open(cal_file)[0]
                        if hdu.header['FILTER'].strip() == band:
                            os.system('cp %s %s/' % (cal_file, input_dir))

                # Run lv3 asn generation
                asn_file = self.run_asn3(directory=input_dir,
                                         band=band)

                # Run pipeline
                self.run_pipeline(band=band,
                                  input_dir=input_dir,
                                  output_dir=output_dir,
                                  asn_file=asn_file,
                                  pipeline_stage='lev3')

            if self.do_astrometric_alignment:
                input_dir = os.path.join(self.reprocess_dir,
                                         self.galaxy,
                                         band,
                                         'lev3')

                self.astrometric_align(input_dir=input_dir)

    def run_destripe(self,
                     files,
                     output_dir,
                     ):
        """Run destriping algorithm, looping over calibrated files

        Args:
            * files (list): List of files to loop over
            * output_dir (str): Where to save destriped files to
        """

        for in_file in files:

            out_file = os.path.join(output_dir,
                                    os.path.split(in_file)[-1])

            median_filter_scales = [7, 31, 63, 127, 511]

            # Run destriping. TODO: Swap out for newer algorithm at some point
            if not os.path.exists(out_file) or self.overwrite_destriping:
                nc_destripe = NircamDestriper(hdu_name=in_file,
                                              hdu_out_name=out_file,
                                              destriping_method='median_filter',
                                              median_filter_scales=median_filter_scales,
                                              quadrants=False,
                                              )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    nc_destripe.run_destriping()

    def run_asn2(self,
                 directory=None,
                 band=None,
                 ):
        """Setup asn lev2 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
        """

        if directory is None:
            raise Warning('Directory should be specified!')
        if band is None:
            raise Warning('Band should be specified!')

        orig_dir = os.getcwd()

        os.chdir(directory)

        asn_lev2_filename = 'asn_lev2_%s.json' % band

        if not os.path.exists(asn_lev2_filename) or self.overwrite_lv2:

            tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', "Program"],
                        dtype=[str, str, str, str, str, float, str, str])

            all_fits_files = glob.glob('*nrc*_rate.fits')
            for f in all_fits_files:
                tab.add_row(parse_fits_to_table(f))
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
            for row in tab:
                json_content['products'].append({
                    "name": os.path.split(row['File'])[1].split('_rate.fits')[0],
                    "members": [
                        {'expname': row['File'],
                         'exptype': 'science',
                         'exposerr': 'null'}
                    ]
                })
            with open(asn_lev2_filename, 'w') as f:
                json.dump(json_content, f)

        os.chdir(orig_dir)

        return asn_lev2_filename

    def run_asn3(self,
                 directory=None,
                 band=None):
        """Setup asn lev3 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
        """

        if band is None:
            raise Warning('Band must be specified!')

        orig_dir = os.getcwd()

        os.chdir(directory)

        asn_lev3_filename = 'asn_lev3_%s.json' % band

        if not os.path.exists(asn_lev3_filename) or self.overwrite_lv3:

            lev2_files = glob.glob('*_cal.fits')
            tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', 'Program'],
                        dtype=[str, str, str, str, str, float, str, str])

            for f in lev2_files:
                tab.add_row(parse_fits_to_table(f))

            json_content = {"asn_type": "None",
                            "asn_rule": "DMS_Level3_Base",
                            "version_id": time.strftime('%Y%m%dt%H%M%S'),
                            "code_version": jwst.__version__,
                            "degraded_status": "No known degraded exposures in association.",
                            "program": tab['Program'][0],
                            "constraints": "No constraints",
                            "asn_id": 'o' + tab['Obs_ID'][0],
                            "asn_pool": "none",
                            "products": [{'name': '%s_nircam_lvl3_%s' % (self.galaxy.lower(), band.lower()),
                                          'members': []}]
                            }
            for row in tab:
                json_content['products'][-1]['members'].append(
                    {'expname': row['File'],
                     'exptype': 'science',
                     'exposerr': 'null'}
                )

            with open(asn_lev3_filename, 'w') as f:
                json.dump(json_content, f)

        os.chdir(orig_dir)

        return asn_lev3_filename

    def run_pipeline(self,
                     band,
                     input_dir,
                     output_dir,
                     asn_file,
                     pipeline_stage,
                     ):
        """Run NIRCAM pipeline.

        Args:
            * band (str): JWST filter
            * input_dir (str): Files associated to asn_file
            * output_dir (str): Where to save the pipeline outputs
            * asn_file (str): Path to asn file. For level1, set this to None
            * pipeline_stage (str): Pipeline processing stage. Should be 'lev1', 'lev2', or 'lev3'
        """

        orig_dir = os.getcwd()

        os.chdir(input_dir)

        if pipeline_stage == 'lev1':

            if self.overwrite_lv1:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or self.overwrite_lv1:

                uncal_files = glob.glob('*_uncal.fits'
                                        )

                uncal_files.sort()

                for uncal_file in uncal_files:

                    detector1 = calwebb_detector1.Detector1Pipeline()

                    # Set some parameters that pertain to the
                    # entire pipeline
                    detector1.output_dir = output_dir
                    if not os.path.isdir(detector1.output_dir):
                        os.makedirs(detector1.output_dir)
                    detector1.save_results = True

                    # Don't flag everything if only the first sample is saturated
                    detector1.jump.suppress_one_group = False
                    detector1.ramp_fit.suppress_one_group = False

                    # Tweak settings
                    detector1.refpix.use_side_ref_pixels = True
                    # detector1.ramp_fit.save_results = True
                    # detector1.linearity.save_results = True

                    # Pull out the trapsfilled file from preceding exposure if needed
                    uncal_file_split = uncal_file.split('_')
                    exposure_str = uncal_file_split[2]
                    exposure_int = int(exposure_str)

                    if exposure_int > 1:
                        previous_exposure_str = '%05d' % (exposure_int - 1)
                        persist_fits_file = uncal_file.replace(exposure_str, previous_exposure_str)
                        persist_fits_file = persist_fits_file.replace('_uncal.fits', '_trapsfilled.fits')

                        persist_file = glob.glob(os.path.join(self.raw_dir,
                                                              self.galaxy,
                                                              'mastDownload',
                                                              'JWST',
                                                              '*nrc*',
                                                              persist_fits_file,
                                                              )
                                                 )[0]
                    else:
                        persist_file = ''

                    # Specify the name of the trapsfilled file
                    detector1.persistence.input_trapsfilled = persist_file

                    # Run the level 1 pipeline
                    detector1.run(uncal_file)

        elif pipeline_stage == 'lev2':

            if self.overwrite_lv2:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or self.overwrite_lv2:

                os.system('rm -rf %s' % output_dir)

                nircam_im2 = calwebb_image2.Image2Pipeline()
                nircam_im2.output_dir = output_dir
                if not os.path.isdir(nircam_im2.output_dir):
                    os.makedirs(nircam_im2.output_dir)
                nircam_im2.save_results = True

                # Any settings to tweak go here

                # Run the level 2 pipeline
                nircam_im2.run(asn_file)

        elif pipeline_stage == 'lev3':

            if self.overwrite_lv3:
                os.system('rm -rf %s' % output_dir)

            output_fits = '%s_nircam_lvl3_%s_i2d.fits' % (self.galaxy.lower(), band.lower())
            output_file = os.path.join(output_dir, output_fits)

            if not os.path.exists(output_file) or self.overwrite_lv3:

                os.system('rm -rf %s' % output_dir)

                nircam_im3 = calwebb_image3.Image3Pipeline()
                nircam_im3.output_dir = output_dir
                if not os.path.isdir(nircam_im3.output_dir):
                    os.makedirs(nircam_im3.output_dir)
                nircam_im3.save_results = True

                # Alignment settings edited to roughly match the HST setup
                # nircam_im3.tweakreg.snr_threshold = 5.0  # 5.0 the default
                nircam_im3.tweakreg.snr_threshold = 20.0  # 5.0 the default
                nircam_im3.tweakreg.kernel_fwhm = 2.5  # 2.5 is default
                nircam_im3.tweakreg.separation = 0.0  # 1.0 is default
                nircam_im3.tweakreg.brightest = 200  # 100 is default
                # nircam_im3.tweakreg.minobj = 10  # 15 is default
                nircam_im3.tweakreg.minobj = 1  # 15 is default
                nircam_im3.tweakreg.searchrad = 10  # 2.0 is default
                nircam_im3.tweakreg.expand_refcat = True  # False is the default
                # nircam_im3.tweakreg.fitgeometry = 'general'  # rshift is the default
                # nircam_im3.tweakreg.align_to_gaia = True  # False is the default

                # Background matching settings
                nircam_im3.skymatch.skymethod = 'global+match'  # 'match' is the default
                nircam_im3.skymatch.subtract = True  # False is the default

                nircam_im3.skymatch.skystat = 'median'  # mode is the default
                nircam_im3.skymatch.nclip = 20  # 5 is the default
                nircam_im3.skymatch.lsigma = 3  # 4 is the default
                nircam_im3.skymatch.usigma = 3  # 4 is the default

                # Source catalogue settings
                nircam_im3.source_catalog.kernel_fwhm = 2.5  # pixels
                nircam_im3.source_catalog.snr_threshold = 10.

                # Run the level 3 pipeline
                nircam_im3.run(asn_file)

        else:

            raise Warning('Pipeline stage %s not recognised!' % pipeline_stage)

        os.chdir(orig_dir)

    def astrometric_align(self,
                          input_dir):
        """Align JWST image to external .fits image. Probably an HST one"""

        if not self.astrometric_alignment_image:
            raise Warning('astrometric_alignment_image should be set!')

        jwst_files = glob.glob(os.path.join(input_dir,
                                            '*_i2d.fits'))
        if len(jwst_files) == 0:
            raise Warning('No JWST image found')

        ref_hdu = fits.open(self.astrometric_alignment_image)

        ref_data = copy.deepcopy(ref_hdu[0].data)
        ref_data[ref_data == 0] = np.nan

        # Find sources in the input image

        source_cat_name = self.astrometric_alignment_image.replace('.fits', '_src_cat.fits')

        if not os.path.exists(source_cat_name) or self.overwrite_astrometric_alignment:

            mean, median, std = sigma_clipped_stats(ref_data, sigma=3)
            daofind = DAOStarFinder(fwhm=2.5, threshold=20 * std)
            sources = daofind(ref_data - median)
            sources.write(source_cat_name, overwrite=True)

        else:

            sources = QTable.read(source_cat_name)

        # Convert sources into a reference catalogue
        wcs_ref = HSTWCS(ref_hdu, 0)

        ref_tab = Table()
        ref_ra, ref_dec = wcs_ref.all_pix2world(sources['xcentroid'], sources['ycentroid'], 1)

        ref_tab['RA'] = ref_ra
        ref_tab['DEC'] = ref_dec
        ref_tab['x'] = sources['xcentroid']
        ref_tab['y'] = sources['ycentroid']

        for jwst_file in jwst_files:

            aligned_hdu_name = jwst_file.replace('.fits', '_align.fits')

            if not os.path.exists(aligned_hdu_name) or self.overwrite_astrometric_alignment:

                jwst_hdu = fits.open(jwst_file)

                jwst_data = copy.deepcopy(jwst_hdu['SCI'].data)
                jwst_data[jwst_data == 0] = np.nan

                # Find sources in the input image

                source_cat_name = jwst_file.replace('.fits', '_src_cat.fits')

                if not os.path.exists(source_cat_name) or self.overwrite_astrometric_alignment:

                    mean, median, std = sigma_clipped_stats(jwst_data, sigma=3)
                    daofind = DAOStarFinder(fwhm=2.5, threshold=20 * std)
                    sources = daofind(jwst_data - median)
                    sources.write(source_cat_name, overwrite=True)

                else:

                    sources = QTable.read(source_cat_name)

                # Convert sources into a reference catalogue
                wcs_jwst = HSTWCS(jwst_hdu, 'SCI')

                jwst_tab = Table()
                jwst_ra, jwst_dec = wcs_jwst.all_pix2world(sources['xcentroid'], sources['ycentroid'], 1)

                jwst_tab['RA'] = jwst_ra
                jwst_tab['DEC'] = jwst_dec
                jwst_tab['x'] = sources['xcentroid']
                jwst_tab['y'] = sources['ycentroid']

                # And match!
                match = TPMatch(searchrad=100,
                                separation=0.1,
                                tolerance=3,
                                use2dhist=True,
                                )
                wcs_jwst_corrector = FITSWCS(wcs_jwst)
                ref_idx, jwst_idx = match(ref_tab, jwst_tab, wcs_jwst_corrector)

                # Finally, do the alignment
                wcs_aligned = fit_wcs(ref_tab[ref_idx], jwst_tab[jwst_idx], wcs_jwst_corrector).wcs

                updatehdr.update_wcs(jwst_hdu,
                                     'SCI',
                                     wcs_aligned,
                                     wcsname='TWEAK',
                                     reusename=True,
                                     verbose=True)

                jwst_hdu.writeto(aligned_hdu_name, overwrite=True)
