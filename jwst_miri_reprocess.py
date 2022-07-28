from astropy.io import fits
import json
import os
import glob
import shutil
import numpy as np
import time
from astropy.table import Table
import warnings
from astropy.modeling import models, fitting
from photutils import make_source_mask

os.environ["CRDS_SERVER_URL"] = "https://jwst-crds-pub.stsci.edu"

# !!! Set up your path to the directory where the necessary calibration files for the pipeline will be stored !!!
os.environ["CRDS_PATH"] = "/home/egorov/Science/PHANGS/JWST/"
import jwst
from jwst.pipeline import calwebb_image2, calwebb_image3


def check_json_kw(file, kw, param):
    if param is None:
        return True
    with open(file) as f:
        data_js = json.load(f)
        if (kw not in data_js) or (param != data_js[kw]):
            return False
        else:
            return True


def parse_fits_to_table(file, check_type=True, suffix_in='_rate.fits'):
    if check_type:
        if file.split(suffix_in)[0].split('_')[-3][2] == '1':
            f_type = 'sci'
        else:
            f_type = 'bgr'
    else:
        f_type = 'sci'
    with fits.open(file) as hdu:
        return file, f_type, hdu[0].header['OBSERVTN'], hdu[0].header['filter'], \
               hdu[0].header['DATE-BEG'], hdu[0].header['DURATION'], \
               hdu[0].header['OBSLABEL'].lower().strip(), hdu[0].header['PROGRAM']


def check_ids(tab, kw, param, alt_rule=None,  add_rule=None, reverse=False):
    if alt_rule is None:
        alt_rule = np.zeros(len(tab), dtype=bool)
    if add_rule is None:
        add_rule = np.ones(len(tab), dtype=bool)
    rec = np.ones(len(tab), dtype=bool)
    for p in np.atleast_1d(param):
        if kw == 'Objname':
            rec1 = p.lower() in tab[kw]
        else:
            if p.startswith('o') or p.startswith('c'):
                p = p[1:]
            rec1 = tab[kw] == p
        if reverse:
            rec1 = ~rec1
        rec = rec & rec1
    return (rec & add_rule) | alt_rule


def tweak_fits(data_dir=None, suffix_in='_cal.fits', suffix_out='_cal_mod.fits', level=3, max_bgr_level=0.3):
    """
    Adjust the content of fits files before further processing and saves the modified version to the same directory.
    Currently includes:
        - Masking the coronograph region
    :param data_dir: direcory with the fits files to modify (absolute path)
    :param suffix_in: Endings of all the files to be changed to suffix_out
    :param suffix_out: New endings of the output files
    :param level: current level of data reduction (2 or 3, but currently only 3 makes sense)
    :param max_bgr_level: maximal level of the pixel to be considered as the background for lyot adjustments
    """
    if data_dir is None:
        return
    if level == 2:
        all_fits_files = [os.path.join(d[0], f) for d in os.walk(data_dir, followlinks=True) if
                          'mirimage' in d[0] for f in d[2] if suffix_in in f]
    else:
        all_fits_files = [os.path.join(d[0], f) for d in os.walk(data_dir, followlinks=True)
                          for f in d[2] if suffix_in in f]
    for f in all_fits_files:
        with fits.open(f) as hdu:
            lyot = hdu[1].data[750:, :280]
            image = hdu[1].data[:, 360:]
            bgr_level_image = np.nanmedian(model_background(image, poly_order=0, sigma=2.,
                                                            gal_mask=(image == 0) |
                                                                     (image > max_bgr_level) | (image < -10.)))
            bgr_level_lyot = np.nanmedian(model_background(lyot, poly_order=0, sigma=2.,
                                                           gal_mask=(lyot == 0) |
                                                                    (lyot > max_bgr_level) | (lyot < -10.)))
            hdu[1].data[750:, :280] += (bgr_level_image-bgr_level_lyot)
            # hdu['DQ'].data[-744:, :280] = 513  # Masks the coronograph area
            hdu.writeto(f.split(suffix_in)[0]+suffix_out, overwrite=True)


def run_association_lev2(data_dir=None, fileout=None, force_id=None, skip_id=None,
                         skip_bgr_num=None, suffix_in='_rate.fits'):
    """
    Create json file with the association of science and background MIRI exposures based
    from the downloaded level2 data
    :param data_dir: Path to the root directory with the downloaded level2 data
    :param fileout: Name of the output json file
    :param force_id: if not None (default), then contain dictionary with keys 'sci_id', 'bgr_id', 'obj_name'.
        If the values are not None, then only those IDs for science, offset exposures or target will be used
    :param skip_id: if not None (default), then contain dictionary with keys 'sci_id', 'bgr_id', 'obj_name'.
        If the values are not None, then those IDs for science, offset exposures or target will be skipped
    :param skip_bgr_num: if not None (default), then should list the numbers of background exposures in each filter,
        which will not be used for combined background (e.g. skip_bgr_num=[0] will skip the first off after sci exp.)
    :param suffix_in: ending of the fits files to be processed (default = '_rate.fits')
    :return: table with the parsed info about the data in the data_dir
    """
    tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', "Program"],
                dtype=[str, str, str, str, str, float, str, str])

    all_fits_files = [os.path.join(d[0], f) for d in os.walk(data_dir, followlinks=True) if
                      'mirimage' in d[0] for f in d[2] if suffix_in in f]
    for f in all_fits_files:
        tab.add_row(parse_fits_to_table(f, suffix_in=suffix_in))
    tab.sort(keys='Start')
    rec = np.ones(shape=len(tab), dtype=bool)
    if force_id is not None:
        for key in ['sci_id', 'bgr_id']:
            if force_id[key] is not None:
                rec = rec & check_ids(tab, 'Obs_ID', force_id[key], add_rule=tab['Type'] == key[:3],
                                      alt_rule=tab['Type'] != key[:3])
        if force_id['obj_name'] is not None:
            rec = rec & check_ids(tab, 'Objname', force_id['obj_name'])
    if skip_id is not None:
        for key in ['sci_id', 'bgr_id']:
            if skip_id[key] is not None:
                rec = rec & check_ids(tab, 'Obs_ID', skip_id[key], add_rule=tab['Type'] == key[:3],
                                      alt_rule=tab['Type'] != key[:3], reverse=True)
        if skip_id['obj_name'] is not None:
            rec = rec & check_ids(tab, 'Objname', skip_id['obj_name'], reverse=True)
    tab = tab[rec]
    n_sci = len(tab['Type'] == 'sci')
    n_bgr = len(tab['Type'] == 'bgr')
    if (n_sci == 0) or (n_bgr == 0):
        print("Error: either science or offset exposures were not found. There is no sense to proceed further.")
        return None
    if fileout is not None:
        t_sci = tab[tab['Type'] == 'sci']
        t_bgr = tab[tab['Type'] == 'bgr']
        json_content = {"asn_type": "image2",
                        "asn_rule": "DMSLevel2bBase",
                        "version_id": time.strftime('%Y%m%dt%H%M%S'),
                        "code_version": jwst.__version__,
                        "degraded_status": "No known degraded exposures in association.",
                        "program": t_sci['Program'][0],
                        "constraints": "none",
                        "asn_id": 'o'+(t_sci['Obs_ID'][0]),
                        "asn_pool": "none",
                        "products": []
                        }
        for t_row in t_sci:
            # if t_row['Filter'] != 'F2100W':
            #     continue
            rec = t_bgr['Filter'] == t_row['Filter']
            if np.sum(rec) == 0:
                continue
            json_content['products'].append({
                "name": os.path.split(t_row['File'])[1].split(suffix_in)[0],
                "members": [
                    {'expname': t_row['File'],
                     'exptype': 'science',
                     'exposerr': 'null'}
                ]
            })
            for bgr_num, t_row_bgr in enumerate(t_bgr[rec]):
                if skip_bgr_num is not None:
                    if bgr_num in np.atleast_1d(skip_bgr_num):
                        continue
                json_content['products'][-1]['members'].append(
                    {'expname': t_row_bgr['File'],
                     'exptype': 'background',
                     'exposerr': 'null'}
                )

        with open(fileout, 'w') as f:
            json.dump(json_content, f)
    return tab


def run_association_lev3(data_dir=os.path.curdir, fileout_root='asn_lev3', galname='', suffix_in='mirimage_cal.fits'):
    os.chdir(data_dir)
    lev2_files = glob.glob(f'*{suffix_in}')
    tab = Table(names=['File', 'Type', 'Obs_ID', 'Filter', 'Start', 'Exptime', 'Objname', "Program"],
                dtype=[str, str, str, str, str, float, str, str])

    for f in lev2_files:
        tab.add_row(parse_fits_to_table(f, suffix_in=suffix_in))
    all_filters = np.unique(tab['Filter'])
    for cur_filter in all_filters:
        cur_asn_lev3 = os.path.join(data_dir, f'{fileout_root}_{cur_filter}.json')
        json_content = {"asn_type": "None",
                        "asn_rule": "DMS_Level3_Base",
                        "version_id": time.strftime('%Y%m%dt%H%M%S'),
                        "code_version": jwst.__version__,
                        "degraded_status": "No known degraded exposures in association.",
                        "program": (tab['Program'][tab['Filter'] == cur_filter])[0],
                        "constraints": "No constraints",
                        "asn_id": 'o' + (tab['Obs_ID'][tab['Filter'] == cur_filter])[0],
                        "asn_pool": "none",
                        "products": [{'name': f'{galname.lower()}_miri_lvl3_{cur_filter.lower()}',
                                      'members': []}]
                        }
        for t_row in tab[tab['Filter'] == cur_filter]:
            # if cur_filter == 'F2100W':
            #     with fits.open(t_row['File']) as hdu:
            #         hdu['DQ'].data[-744:, :280] = 513
            #         # data_analyse_for_bgr = hdu[1].data
            #         # data_analyse_for_bgr[-744:, :280] = 0
            #         # bgr = model_background(data_analyse_for_bgr, poly_order=0,
            #         #                        gal_mask=(data_analyse_for_bgr == 0) | (data_analyse_for_bgr >= .15),
            #         #                        filter_fwhm=3.)
            #         # hdu[1].data[data_analyse_for_bgr != 0] -= bgr[data_analyse_for_bgr != 0]
            #         hdu.writeto(t_row['File'], overwrite=True)
            json_content['products'][-1]['members'].append(
                {'expname': t_row['File'],
                 'exptype': 'science',
                 'exposerr': 'null'}
            )

        with open(cur_asn_lev3, 'w') as f:
            json.dump(json_content, f)


def run_pipeline(asn_file, level='level2', data_dir=os.curdir, output_dir=os.curdir, max_bgr_level=0.3):
    """
    Run JWST pipeline for the selected stage of the data reduction and for the provided asn file
    :param asn_file: path to the asn json file to be used for the current data processing
    :param level: stage of the data processing ('level2' or 'level3')
    :param data_dir: path to the directory with the data to be processed
    :param output_dir: path to the directory where all output files will be saved
    :param max_bgr_level: maximal brightness of the pixels considered as sky for skymatch (level3 only)
    """

    if level == 'level2':
        image2 = calwebb_image2.Image2Pipeline()
        # For some reason, setting up the input directory hadn't worked, so just cd to the level2 directory
        os.chdir(data_dir)
        image2.output_dir = output_dir
        image2.save_results = True
        image2.resample.pixfrac = 1.0
        image2.bkg_subtract.save_combined_background = True
        # image2.save_bsub = True
        image2.bkg_subtract.sigma = 1.5
        image2.run(asn_file)

    elif level == 'level3':
        miri_im3 = calwebb_image3.Image3Pipeline()
        miri_im3.output_dir = output_dir
        if not os.path.isdir(miri_im3.output_dir):
            os.mkdir(miri_im3.output_dir)
        miri_im3.save_results = True
        # Set some parameters for individual steps.# HINT: the PSF FWHM for MIRI with the F700W filter# is 2.187 pixels.
        miri_im3.tweakreg.snr_threshold = 5.0  # 5.0 is the default
        if "F770W" in asn_file:
            miri_im3.tweakreg.kernel_fwhm = 2.187  # 2.5 is the default
            miri_im3.source_catalog.kernel_fwhm = 2.187  # pixels
        else:
            miri_im3.tweakreg.kernel_fwhm = 2.5  # 2.5 is the default
            miri_im3.source_catalog.kernel_fwhm = 2.5  # pixels
        miri_im3.tweakreg.brightest = 10  # 100 is the default

        miri_im3.source_catalog.snr_threshold = 5.
        # miri_im3.skymatch.skip = False
        miri_im3.skymatch.skymethod = 'global+match'
        miri_im3.skymatch.upper = max_bgr_level
        miri_im3.skymatch.lower = -5.
        miri_im3.skymatch.nclip = 10
        miri_im3.skymatch.subtract = False
        # miri_im3.skymatch.match_down = True
        miri_im3.skymatch.usigma = 1.5
        miri_im3.skymatch.lsigma = 1.5
        miri_im3.run(asn_file)
    else:
        print("ERROR: Unknown level of data processing.")


def model_background(data, poly_order=1, sigma=2, n_pixels=5, pixel_step=1, gal_mask=None, **kwargs):
    """Model the background as a 2D polynomial."""

    y, x = np.mgrid[:data.shape[0], :data.shape[1]]    # Create a mask for the data.

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        mask = make_source_mask(data, nsigma=sigma, npixels=n_pixels, mask=gal_mask, **kwargs)

    data_masked = data.copy()
    data_masked[mask] = np.nan

    if gal_mask is not None:
        data_masked[gal_mask] = np.nan
    # Remove any NaNs (i.e. data missing or data masked).
    not_nan = np.where(~np.isnan(data_masked))
    nan_idx = np.where(np.isnan(data))
    x_fit = x[not_nan]
    y_fit = y[not_nan]
    data_fit = data[not_nan]

    # Fit background to the data.
    p_init = models.Polynomial2D(poly_order)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter.
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x_fit[::pixel_step], y_fit[::pixel_step], data_fit[::pixel_step])

    background = p(x, y)
    background[nan_idx] = np.nan
    return background


def run_reprocessing(galname="ngc0628", lev2_root_dir='/home/egorov/Science/PHANGS/JWST/',
                     reduced_root_dir="/home/egorov/Science/PHANGS/JWST/Reduction/",
                     asn_lev2_rules=None, do_steps=None,
                     do_adjust_lyot=True, max_bgr_level=0.3):

    # ===========================
    # We assume that the level2 and the output directories contain subfolders for each galaxy.
    # If not - then correct two lines below. Otherwise, no any modification required
    # ===========================

    if do_steps is None:
        do_steps = {
            'asn2': True,
            'process2': True,
            'mask_level3': True,
            'asn3': True,
            'process3': True
        }

    lev2_dir_cur_obj = os.path.join(lev2_root_dir, galname)
    reduced_dir_cur_obj = os.path.join(reduced_root_dir, galname)
    if not os.path.isdir(reduced_dir_cur_obj):
        os.makedirs(reduced_dir_cur_obj)
    asn_file_lev2 = os.path.join(reduced_dir_cur_obj, f'{galname}_asn_lev2.json')
    asn_file_lev3_root = f'{galname}_asn_lev3'

    lev2_suffix_in = '_rate.fits'
    if do_adjust_lyot:
        lev3_suffix_in = 'mirimage_cal_masked.fits'
    else:
        lev3_suffix_in = 'mirimage_cal.fits'

    # === Step 1: Create association file for level2 reprocessing
    if do_steps['asn2']:
        tab = run_association_lev2(lev2_dir_cur_obj, asn_file_lev2,
                                   force_id=asn_lev2_rules['force_use'], skip_id=asn_lev2_rules['skip'],
                                   skip_bgr_num=asn_lev2_rules['skip_bgr_exposures'], suffix_in=lev2_suffix_in)

    # === Step 2: Run level2 data reduction
    if os.path.isfile(asn_file_lev2) and do_steps['process2']:
        run_pipeline(asn_file_lev2, level='level2', data_dir=lev2_dir_cur_obj, output_dir=reduced_dir_cur_obj)

    # === Step 3: Mask the input frames before level2 processing
    if do_adjust_lyot and do_steps['mask_level3']:
        tweak_fits(reduced_dir_cur_obj, suffix_in='mirimage_cal.fits', suffix_out=lev3_suffix_in, level=3,
                   max_bgr_level=max_bgr_level)

    # === Step 4: Create association file for level3 reprocessing
    if do_steps['asn3']:
        run_association_lev3(reduced_dir_cur_obj, fileout_root=asn_file_lev3_root, galname=galname,
                             suffix_in=lev3_suffix_in)

    # === Step 5: Run level3 data reduction
    if do_steps['process3']:
        os.chdir(reduced_dir_cur_obj)
        asn3_json_files = [f for f in glob.glob('*.json') if asn_file_lev3_root in f]
        for asn_f in asn3_json_files:
            cur_filter = asn_f.split('.json')[0].split('_')[-1]
            cur_dir = os.path.join(reduced_dir_cur_obj, cur_filter)
            run_pipeline(asn_f, level='level3', data_dir=reduced_dir_cur_obj, output_dir=cur_dir,
                         max_bgr_level=max_bgr_level)


if __name__ == '__main__':
    # To run reprocessing, it is necessary to provide:
    # 1) name of the galaxy;
    # 2) path to the directory where the level2 data were downloaded from MAST
    # 3) path to the directory where the reprocessed files will be saved, together with json association files
    # 4) Only if needed - adjust the dictionary for the rules to process the files association
    # 5) Also can be useful to modify do_adjust_lyot and max_bgr_level for some galaxies (see below)

    galname = "ngc0628"
    lev2_root_dir = '/home/egorov/Science/PHANGS/JWST/'
    reduced_root_dir = "/home/egorov/Science/PHANGS/JWST/Reduction/"

    # This values can be adjusted depending on the galaxy.
    # Current recomendation
    #   - use do_adjust_lyot = True and max_bgr_level = 0.3 for IC5332 for NGC628 and NGC7496;
    #   - use do_adjust_lyot = False and max_bgr_level = 5.5 for IC5332
    do_adjust_lyot = True  # Set True if you want to adjust background level in Lyot area (for proper sky matching)
    max_bgr_level = 0.3  # Maximal brightness in calibrated images to be considered as sky (for proper sky matching)

    asn_lev2_rules = {
        "force_use": {"sci_id": None, 'bgr_id': None, 'obj_name': None},  # if not None -> only these ids will be used
        "skip": {"sci_id": None, 'bgr_id': None, 'obj_name': None},  # if not None -> these ids will be skipped
        "skip_bgr_exposures": None,  # if not None -> these off exposures in each filter will be skipped
                                       # (e.g., [0] will skip the first off image after science exposures)
    }

    # Select steps:
    do_steps = {
        'asn2': True,
        'process2': True,
        'mask_level3': True,
        'asn3': True,
        'process3': True,
    }

    run_reprocessing(galname=galname, lev2_root_dir=lev2_root_dir,
                     reduced_root_dir=reduced_root_dir, asn_lev2_rules=asn_lev2_rules, do_steps=do_steps,
                     do_adjust_lyot=do_adjust_lyot, max_bgr_level=max_bgr_level)


