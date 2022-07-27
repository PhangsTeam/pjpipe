import glob
import json
import os
import time
import warnings

from astropy.io import fits
from astropy.table import Table

from nircam_destriping import NircamDestriper

jwst = None
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
                 lv2_root_dir,
                 working_dir,
                 bands=None,
                 do_steps=None,
                 overwrite=False,
                 crds_url='https://jwst-crds-pub.stsci.edu'
                 ):
        """NIRCAM reprocessing routines.

        Runs destriping algorithm on calibrated data before processing to level 3 mosaics. Has some tweaks in the
        level 3 processing to work better for the PHANGS data

        Args:
            * crds_path (str): Path to CRDS data
            * galaxy (str): Galaxy to run reprocessing for
            * lv2_root_dir (str): Path to raw data
            * working_dir (str): Path to reprocess data into
            * bands (list): JWST filters to loop over
            * do_steps (dict): steps to run, should be in dict of form 'key': True. Options 'destripe', 'asn3',
                'process3'
            * overwrite (bool): Whether to overwrite destriping/reprocessing. Defaults to False
            * crds_url (str): URL to get CRDS files from. Defaults to 'https://jwst-crds-pub.stsci.edu'
        """

        os.environ['CRDS_SERVER_URL'] = crds_url
        os.environ['CRDS_PATH'] = crds_path

        global jwst
        import jwst

        global calwebb_image3
        from jwst.pipeline import calwebb_image3

        self.galaxy = galaxy
        self.lv2_root_dir = lv2_root_dir
        self.working_dir = working_dir

        if bands is None:
            bands = ['F200W', 'F300M', 'F335M', 'F360M']

        self.bands = bands

        if do_steps is None:
            do_steps = {'destripe': True,
                        'asn3': True,
                        'process3': True,
                        }

        self.do_steps = do_steps

        self.overwrite = overwrite

    def run_all(self):
        """Run the whole pipeline reprocess"""

        lv2_files = glob.glob(os.path.join(self.lv2_root_dir,
                                           self.galaxy,
                                           'mastDownload',
                                           'JWST',
                                           '*nrc*',
                                           '*nrc*_cal.fits')
                              )
        lv2_files.sort()

        for band in self.bands:

            band_lv2_files = []
            for lv2_file in lv2_files:
                hdu = fits.open(lv2_file)[0]
                if hdu.header['FILTER'].strip() == band:
                    band_lv2_files.append(lv2_file)

            if self.do_steps['destripe']:
                self.run_destripe(files=band_lv2_files, band=band)

            if self.do_steps['asn3']:
                self.run_asn3(band=band)

            if self.do_steps['process3']:
                self.run_pipeline(band=band)

    def run_destripe(self,
                     files=None,
                     band=None):
        """Run destriping algorithm, looping over calibrated files

        Args:
            * files (list): List of files to loop over
            * band (str): JWST filter
        """

        if files is None:
            raise Warning('Need files to destripe!')

        if band is None:
            raise Warning('Need a band for the file structure!')

        output_dir = os.path.join(self.working_dir,
                                  self.galaxy,
                                  band)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for in_file in files:

            out_file = os.path.join(output_dir,
                                    os.path.split(in_file)[-1])

            median_filter_scales = [7, 31, 63, 127, 511]

            # Run destriping. TODO: Swap out for newer algorithm at some point
            if not os.path.exists(out_file) or self.overwrite:
                nc_destripe = NircamDestriper(hdu_name=in_file,
                                              hdu_out_name=out_file,
                                              destriping_method='median_filter',
                                              median_filter_scales=median_filter_scales,
                                              quadrants=True,
                                              )
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    nc_destripe.run_destriping()

    def run_asn3(self,
                 band=None):
        """Setup asn lev3 files to run on destriped data

        Args:
            * band (str): JWST filter
        """

        if band is None:
            raise Warning('Band must be specified!')

        orig_dir = os.getcwd()

        working_band_dir = os.path.join(self.working_dir,
                                        self.galaxy,
                                        band)

        os.chdir(working_band_dir)

        lev2_files = glob.glob('*_cal.fits')
        tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', 'Program'],
                    dtype=[str, str, str, str, str, float, str, str])

        for f in lev2_files:
            tab.add_row(parse_fits_to_table(f))
        asn_lev3_filename = 'asn_lev3_%s.json' % band
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
        for t_row in tab:
            json_content['products'][-1]['members'].append(
                {'expname': t_row['File'],
                 'exptype': 'science',
                 'exposerr': 'null'}
            )

        with open(asn_lev3_filename, 'w') as f:
            json.dump(json_content, f)

        os.chdir(orig_dir)

    def run_pipeline(self, band):
        """Run level 3 pipeline on destriped data

        Args:
            * band (str): JWST filter
        """

        if band is None:
            raise Warning('Band should be defined!')

        orig_dir = os.getcwd()

        working_band_dir = os.path.join(self.working_dir,
                                        self.galaxy,
                                        band)
        output_dir = os.path.join(self.working_dir,
                                  self.galaxy,
                                  'lev3',
                                  band)
        asn_file = 'asn_lev3_%s.json' % band

        os.chdir(working_band_dir)

        output_fits = '%s_nircam_lvl3_%s_i2d.fits' % (self.galaxy.lower(), band.lower())
        output_file = os.path.join(output_dir, output_fits)

        if not os.path.exists(output_file) or self.overwrite:

            os.system('rm -rf %s' % output_dir)

            nircam_im3 = calwebb_image3.Image3Pipeline()
            nircam_im3.output_dir = output_dir
            if not os.path.isdir(nircam_im3.output_dir):
                os.makedirs(nircam_im3.output_dir)
            nircam_im3.save_results = True

            # Alignment settings
            nircam_im3.tweakreg.snr_threshold = 5.0  # 5.0 is the default
            nircam_im3.tweakreg.kernel_fwhm = 2.5  # 2.5 is the default
            nircam_im3.tweakreg.brightest = 200  # 100 is the default
            nircam_im3.tweakreg.minobj = 10  # 15 is default
            nircam_im3.tweakreg.expand_refcat = True  # False is the default
            nircam_im3.tweakreg.fitgeometry = 'general'  # rshift is the default
            nircam_im3.tweakreg.align_to_gaia = True  # False is the default

            # Background matching settings
            nircam_im3.skymatch.skymethod = 'global+match'
            nircam_im3.skymatch.subtract = True  # False is the default

            # Source catalogue settings
            nircam_im3.source_catalog.kernel_fwhm = 2.5  # pixels
            nircam_im3.source_catalog.snr_threshold = 10.

            # Run the level 3 pipeline
            nircam_im3.run(asn_file)

        os.chdir(orig_dir)
