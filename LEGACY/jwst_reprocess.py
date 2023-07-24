import copy
import functools
import gc
import glob
import inspect
import itertools
import json
import logging
import multiprocessing as mp
import os
import random
import shutil
import time
import warnings
from functools import partial
from multiprocessing import cpu_count

import astropy.units as u
import gwcs
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata.bitmask import interpret_bit_flags, bitfield_to_boolean_mask
from astropy.stats import sigma_clipped_stats, SigmaClip
from astropy.table import Table, QTable
from astropy.wcs import WCS
from image_registration import cross_correlation_shifts
from jwst import datamodels
from jwst.assign_wcs.util import update_fits_wcsinfo
from jwst.datamodels.dqflags import pixel
from photutils.detection import DAOStarFinder
from photutils.segmentation import detect_threshold, detect_sources
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
from reproject.mosaicking.subset_array import ReprojectedArraySubset
from scipy.ndimage import uniform_filter1d
from stwcs.wcsutil import HSTWCS
from threadpoolctl import threadpool_limits
from tqdm import tqdm
from tweakwcs import fit_wcs, XYXYMatch
from tweakwcs.correctors import FITSWCSCorrector, JWSTWCSCorrector

from nircam_destriping import NircamDestriper

jwst = None
calwebb_detector1 = None
calwebb_image2 = None
calwebb_image3 = None
TweakRegStep = None
SkyMatchStep = None
SourceCatalogStep = None
webbpsf = None
PSFSubtraction = None

files_reproj = None
data_reproj = None
weight_reproj = None

# Pipeline steps
ALLOWED_STEPS = [
    'lv1',
    'lv2',
    'destripe',
    'dither_stripe_sub',
    'bg_sub',
    'dither_match',
    'lyot_adjust',
    'wcs_adjust',
    'psf_model',
    'lv3',
    'astrometric_catalog',
    'astrometric_align',
]

# Pipeline steps where we don't want to delete the whole directory
STEP_NO_DEL_DIR = [
    'astrometric_catalog',
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

PIXEL_SCALES = {
    'miri': 0.11,
    'nircam_long': 0.063,
    'nircam_short': 0.031,
}

RAD2ARCSEC = 3600.0 * np.rad2deg(1.0)


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
        with threadpool_limits(limits=1, user_api=None):
            nc_destripe.run_destriping()

    del nc_destripe
    gc.collect()

    return True


def create_subset_reproj_image(data_array,
                               weight_array,
                               subset_idx,
                               min_area_frac=0.5,
                               ):
    """Create a subset average image from a bunch of reprojected ones

    Args:
        data_array: list of reprojected data files
        weight_array: list of reprojected weight files
        subset_idx: index of image to create the subset average image for
        min_area_frac (float): Minimum fraction of overlapping pixels for image
            to be included in the average. Defaults to 0.5 (50%)
    """

    file_data = copy.deepcopy(data_array[subset_idx])
    file_weight = copy.deepcopy(weight_array[subset_idx])

    data = np.zeros_like(file_data.array)
    weights = np.zeros_like(file_weight.array)

    # First, put the original image in
    data += file_weight.array * file_data.array
    weights += file_weight.array

    # Get the number of pixels, so that we can remove any small overlaps
    # later
    valid_idx = np.where(np.logical_and(data != 0,
                                        np.isfinite(data),
                                        )
                         )
    npix_file = len(valid_idx[0])

    file_imin, file_imax = file_data.imin, file_data.imax
    file_jmin, file_jmax = file_data.jmin, file_data.jmax

    for idx in range(len(data_array)):

        if subset_idx == idx:
            continue

        if not data_array[subset_idx].overlaps(data_array[idx]):
            continue

        data_idx = copy.deepcopy(data_array[idx])
        weight_idx = copy.deepcopy(weight_array[idx])

        data_i = data_idx.array
        weight_i = weight_idx.array

        # Before putting these in, we need to make sure they're on the same level.
        # Do this by subtracting the median of the difference for the first file
        diff = data_idx - data_array[subset_idx]
        diff_weight = weight_idx * weight_array[subset_idx]

        diff = diff.array[diff_weight.array > 0]
        diff[diff == 0] = np.nan
        diff = diff[np.isfinite(diff)]

        if len(diff) == 0:
            continue

        # Pull out indices we'll need later
        idx_imin, idx_imax = data_idx.imin, data_idx.imax
        idx_jmin, idx_jmax = data_idx.jmin, data_idx.jmax

        _, diff_med, _ = sigma_clipped_stats(diff, maxiters=10)
        data_i -= diff_med

        imin = max(file_imin, idx_imin)
        imax = min(file_imax, idx_imax)
        jmin = max(file_jmin, idx_jmin)
        jmax = min(file_jmax, idx_jmax)

        if imax < imin:
            imax = imin

        if jmax < jmin:
            jmax = jmin

        final_slice = (slice(jmin - file_jmin, jmax - file_jmin), slice(imin - file_imin, imax - file_imin))
        reproj_slice = (slice(jmin - idx_jmin, jmax - idx_jmin), slice(imin - idx_imin, imax - idx_imin))

        # Put data and weights into array
        new_data = np.zeros_like(data)
        new_weights = np.zeros_like(weights)
        new_data[final_slice] += weight_i[reproj_slice] * data_i[reproj_slice]
        new_weights[final_slice] += weight_i[reproj_slice]

        # Check we have the required amount of overlap
        valid_i = new_data[valid_idx]
        valid_i = valid_i[np.logical_and(valid_i != 0,
                                         np.isfinite(valid_i)
                                         )]

        area_frac = len(valid_i) / npix_file

        if area_frac < min_area_frac:
            continue

        data += new_data
        weights += new_weights

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data_avg = data / weights
    data_avg[data_avg == 0] = np.nan

    return data_avg, file_imin, file_imax, file_jmin, file_jmax


def parallel_per_dither_stripe_sub(dither,
                                   in_dir,
                                   out_dir,
                                   step_ext,
                                   plot_dir,
                                   weight_type='exptime',
                                   **dither_stripe_kws,
                                   ):
    """Wrapper to parallelise up dither stripe subtraction, if we're grouping up dithers"""

    allowed_weight_types = ['exptime', 'ivm']

    if weight_type not in allowed_weight_types:
        raise ValueError('weight_type should be one of %s' % allowed_weight_types)

    dither_files = glob.glob(os.path.join(in_dir,
                                          '%s*_%s.fits' % (dither, step_ext))
                             )

    dither_files.sort()

    results = []

    _, data_reproj, weight_reproj, optimal_wcs = weighted_reproject_image(dither_files,
                                                                          weight_type=weight_type,
                                                                          )

    for file_idx, file in enumerate(dither_files):
        # Create the average image

        data_avg, imin, imax, jmin, jmax = create_subset_reproj_image(data_reproj,
                                                                      weight_reproj,
                                                                      subset_idx=file_idx,
                                                                      )

        data_avg_wcs = optimal_wcs[jmin:jmax, imin:imax]

        result = dither_stripe_sub(file,
                                   data_avg_array=data_avg,
                                   data_avg_wcs=data_avg_wcs,
                                   out_dir=out_dir,
                                   plot_dir=plot_dir,
                                   **dither_stripe_kws,
                                   )
        results.append(result)

    return results


def parallel_dither_stripe_sub(file_idx,
                               optimal_wcs,
                               out_dir,
                               plot_dir,
                               **dither_stripe_kws,
                               ):
    """Function to parallelise up dither subtraction with all info"""

    file = files_reproj[file_idx]

    data_avg, imin, imax, jmin, jmax = create_subset_reproj_image(data_reproj,
                                                                  weight_reproj,
                                                                  subset_idx=file_idx,
                                                                  )

    data_avg_wcs = optimal_wcs[jmin:jmax, imin:imax]

    result = dither_stripe_sub(file,
                               data_avg_array=data_avg,
                               data_avg_wcs=data_avg_wcs,
                               out_dir=out_dir,
                               plot_dir=plot_dir,
                               **dither_stripe_kws,
                               )
    return file_idx, result


def dither_stripe_sub(file,
                      data_avg_array,
                      data_avg_wcs,
                      out_dir,
                      plot_dir,
                      quadrants=True,
                      do_large_scale=True,
                      median_filter_factor=4,
                      sigma=3,
                      dilate_size=7,
                      maxiters=None,
                      ):
    """Do a row-by-row, column-by-column data subtraction using other dither information

    Create a weighted mean image of all overlapping files per-tile, then do a sigma-clipped
    median along columns and rows (optionally by quadrants), and finally a smoothed clip along
    rows after boxcar filtering to remove persistent large-scale ripples in the data

    Args:
        file (str): File to correct
        data_avg_array (np.ndarray): Pre-calculated average image
        data_avg_wcs: WCS for the average image
        out_dir (output): Output directory for the files
        plot_dir (str): Directory to save plots to
        quadrants (bool): Whether to split per-amplifier or not. Defaults to True, but
            will be forced off for subarray data
        do_large_scale (bool): Whether to do boxcar-filtering to try and remove large,
            consistent ripples between data. Defaults to True, but can probably be turned
            off for observations with many overlaps
        median_filter_factor (int): Factor by which we smooth in terms of the array size.
            Defaults to 4, i.e. smoothing scale is 1/4th the array size
        sigma (float): sigma value for sigma-clipped statistics. Defaults to 3
        dilate_size (int): Dilation size for mask creation. Defaults to 7
        maxiters (int): Maximum number of sigma-clipping iterations. Defaults to None

    Returns:
        filename, and stripe model
    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with threadpool_limits(limits=1, user_api=None):

            dq_bits = interpret_bit_flags('~DO_NOT_USE+NON_SCIENCE', flag_name_map=pixel)

            model = datamodels.open(file)
            file_name = model.meta.filename

            if os.path.exists(os.path.join(out_dir, file_name)):
                return

            if 'sub' in model.meta.subarray.name.lower():
                quadrants = False

            dq_bit_mask = bitfield_to_boolean_mask(
                model.dq.astype(np.uint8),
                dq_bits,
                good_mask_value=0,
                dtype=np.uint8
            )

            # Pull out data and DQ mask
            data = copy.deepcopy(model.data)
            data[dq_bit_mask != 0] = np.nan

            wcs = model.meta.wcs.to_fits_sip()

            # Reproject the average image
            data_avg = reproject_interp((data_avg_array, data_avg_wcs),
                                        wcs,
                                        return_footprint=False,
                                        )

        # Replace NaNd data with column median, so the boxcar
        # doesn't catastrophically fail later
        data_avg_col = np.nanmedian(data_avg, axis=1)
        for col in range(data_avg.shape[0]):
            col_idx = np.where(np.isnan(data_avg[col, :]))
            data_avg[col, col_idx[0]] = data_avg_col[col]

        diff_unsmoothed = data - data_avg
        diff_unsmoothed -= np.nanmedian(diff_unsmoothed)

        stripes_arr = np.zeros_like(diff_unsmoothed)

        mask_pos = make_source_mask(diff_unsmoothed,
                                    nsigma=sigma,
                                    dilate_size=dilate_size,
                                    sigclip_iters=maxiters,
                                    )
        mask_neg = make_source_mask(-diff_unsmoothed,
                                    mask=mask_pos,
                                    nsigma=sigma,
                                    dilate_size=dilate_size,
                                    sigclip_iters=maxiters,
                                    )
        mask_unsmoothed = mask_pos | mask_neg  # | dq_bit_mask1

        # First, subtract the y
        stripes_y = sigma_clipped_stats(diff_unsmoothed - stripes_arr,
                                        mask=mask_unsmoothed,
                                        sigma=sigma,
                                        maxiters=maxiters,
                                        axis=0
                                        )[1]

        # Centre around 0, replace NaNs
        stripes_y -= np.nanmedian(stripes_y)
        stripes_y[np.isnan(stripes_y)] = 0

        stripes_arr += stripes_y[np.newaxis, :]

        stripes_x_2d = np.zeros_like(stripes_arr)

        if quadrants:

            quadrant_size = stripes_arr.shape[1] // 4

            for quadrant in range(4):
                idx_slice = slice(quadrant * quadrant_size, (quadrant + 1) * quadrant_size)

                # Do a pass at the unsmoothed data
                diff_quadrants = diff_unsmoothed[:, idx_slice] - stripes_arr[:, idx_slice]
                mask_quadrants = mask_unsmoothed[:, idx_slice]
                stripes_x = sigma_clipped_stats(diff_quadrants,
                                                mask=mask_quadrants,
                                                sigma=sigma,
                                                maxiters=maxiters,
                                                axis=1
                                                )[1]

                # Centre around 0, replace NaNs
                stripes_x -= np.nanmedian(stripes_x)
                stripes_x[np.isnan(stripes_x)] = 0

                stripes_x_2d[:, idx_slice] += stripes_x[:, np.newaxis]

        else:

            # Do a pass at the unsmoothed data
            stripes_x = sigma_clipped_stats(diff_unsmoothed - stripes_arr,
                                            mask=mask_unsmoothed,
                                            sigma=sigma,
                                            maxiters=maxiters,
                                            axis=1
                                            )[1]

            # Centre around 0, replace NaNs
            stripes_x -= np.nanmedian(stripes_x)
            stripes_x[np.isnan(stripes_x)] = 0

            stripes_x_2d += stripes_x[:, np.newaxis]

        # Centre around 0 one last time
        stripes_x_2d -= np.nanmedian(stripes_x_2d)
        stripes_x_2d[np.isnan(stripes_x_2d)] = 0

        stripes_arr += stripes_x_2d
        stripes_arr -= np.nanmedian(stripes_arr)
        stripes_arr[np.isnan(stripes_arr)] = 0

        if do_large_scale:
            # Filter along the y-axis (to preserve the stripe noise) with a large boxcar filter.
            # This ideally flattens out the background, for any consistent large-scale ripples
            # between images
            boxcar = uniform_filter1d(data_avg,
                                      size=data_avg.shape[0] // median_filter_factor,
                                      axis=0,
                                      mode='reflect',
                                      )
            diff_smoothed = data - stripes_arr - boxcar
            diff_smoothed -= np.nanmedian(diff_smoothed)

            mask_pos = make_source_mask(diff_smoothed,
                                        nsigma=sigma,
                                        dilate_size=dilate_size,
                                        sigclip_iters=maxiters,
                                        )
            mask_neg = make_source_mask(-diff_smoothed,
                                        mask=mask_pos,
                                        nsigma=sigma,
                                        dilate_size=dilate_size,
                                        sigclip_iters=maxiters,
                                        )
            mask_smoothed = mask_pos | mask_neg  # | dq_bit_mask1

            # Pass through the smoothed data, but across the whole image to avoid the bright noisy bits
            stripes_x = sigma_clipped_stats(diff_smoothed,
                                            mask=mask_smoothed,
                                            sigma=sigma,
                                            maxiters=maxiters,
                                            axis=1
                                            )[1]

            # Centre around 0, replace NaNs
            stripes_x -= np.nanmedian(stripes_x)
            stripes_x[np.isnan(stripes_x)] = 0

            stripes_arr += stripes_x[:, np.newaxis]

            # Centre around 0 for luck
            stripes_arr -= np.nanmedian(stripes_arr)
            stripes_arr[np.isnan(stripes_arr)] = 0

        # Make diagnostic plot
        plot_name = os.path.join(plot_dir,
                                 file_name.replace('.fits', '_dither_stripe_sub'),
                                 )
        plt.figure(figsize=(9, 6))

        vmin_diff, vmax_diff = np.nanpercentile(diff_unsmoothed, [1, 99])
        vmin_data, vmax_data = np.nanpercentile(data, [5, 95])

        plt.subplot(2, 3, 1)
        plt.imshow(diff_unsmoothed,
                   origin='lower',
                   interpolation='nearest',
                   vmin=vmin_diff,
                   vmax=vmax_diff,
                   )
        plt.axis('off')
        plt.title('Uncorr. diff')

        plt.subplot(2, 3, 2)
        plt.imshow(diff_unsmoothed - stripes_arr,
                   origin='lower',
                   interpolation='nearest',
                   vmin=vmin_diff,
                   vmax=vmax_diff,
                   )
        plt.axis('off')
        plt.title('Corr. diff')

        plt.subplot(2, 3, 3)
        plt.imshow(stripes_arr,
                   origin='lower',
                   interpolation='none',
                   vmin=vmin_diff,
                   vmax=vmax_diff,
                   )
        plt.axis('off')
        plt.title('Stripe model')

        plt.subplot(2, 3, 4)
        plt.imshow(data,
                   origin='lower',
                   interpolation='nearest',
                   vmin=vmin_data,
                   vmax=vmax_data,
                   )
        plt.axis('off')
        plt.title('Uncorr. data')

        plt.subplot(2, 3, 5)
        plt.imshow(data - stripes_arr,
                   origin='lower',
                   interpolation='nearest',
                   vmin=vmin_data,
                   vmax=vmax_data,
                   )
        plt.axis('off')
        plt.title('Corr. data')

        plt.subplots_adjust(wspace=0.01)

        plt.savefig('%s.png' % plot_name, bbox_inches='tight')
        plt.savefig('%s.pdf' % plot_name, bbox_inches='tight')

        plt.close()

    return file, stripes_arr


def parallel_tweakback(crf_file,
                       matrix=None,
                       shift=None,
                       ref_tpwcs=None,
                       ):
    """Wrapper function to parallelise tweakback routine

    """

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if matrix is None:
            matrix = [[1, 0], [0, 1]]
        if shift is None:
            shift = [0, 0]

        crf_input_im = datamodels.open(crf_file)

        crf_wcs = crf_input_im.meta.wcs
        crf_wcsinfo = crf_input_im.meta.wcsinfo.instance

        crf_im = JWSTWCSCorrector(
            wcs=crf_wcs,
            wcsinfo=crf_wcsinfo,
        )
        crf_im.set_correction(matrix=matrix,
                              shift=shift,
                              ref_tpwcs=ref_tpwcs,
                              )

        # crf_input_im = crf_im.meta['image_model']
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

    del crf_im
    del crf_input_im
    gc.collect()

    return True


def make_source_mask(data,
                     mask=None,
                     nsigma=3,
                     npixels=3,
                     dilate_size=11,
                     sigclip_iters=5,
                     ):
    """Make a source mask from segmentation image"""

    sc = SigmaClip(sigma=nsigma,
                   maxiters=sigclip_iters,
                   )
    threshold = detect_threshold(data,
                                 mask=mask,
                                 nsigma=nsigma,
                                 sigma_clip=sc,
                                 )

    segment_map = detect_sources(data,
                                 threshold,
                                 npixels=npixels,
                                 )

    # If sources are detected, we can make a segmentation mask, else fall back to 0 array
    try:
        mask = segment_map.make_source_mask(size=dilate_size)
    except AttributeError:
        mask = np.zeros(data.shape, dtype=bool)

    return mask


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


def background_subtract(hdu,
                        sigma=1.5,
                        n_pixels=5,
                        max_iterations=20,
                        ):
    """Sigma-clipped background subtraction for fits HDU"""

    dq_mask = hdu['DQ'].data != 0

    mean, median, std = sigma_clip(hdu['SCI'].data,
                                   dq_mask=dq_mask,
                                   sigma=sigma,
                                   n_pixels=n_pixels,
                                   max_iterations=max_iterations,
                                   )
    hdu['SCI'].data -= median

    return hdu


def background_match_dithers(dithers_reproj1,
                             dithers_reproj2,
                             plot_name=None,
                             max_points=None,
                             maxiters=10,
                             min_linear_frac=0.25,
                             ):
    """Calculate relative difference between groups of files (i.e. dithers) on the same pixel grid"""

    diffs = []

    if not isinstance(dithers_reproj1, list):
        dithers_reproj1 = [dithers_reproj1]
    if not isinstance(dithers_reproj2, list):
        dithers_reproj2 = [dithers_reproj2]

    n_pix = 0

    fig, axs = None, None
    if plot_name is not None:
        fig, axs = plt.subplots(nrows=len(dithers_reproj1),
                                ncols=len(dithers_reproj2),
                                figsize=(2.5 * len(dithers_reproj2), 2.5 * len(dithers_reproj1)),
                                squeeze=False,
                                )

    lin_size = 0

    for dither_idx1, dither_reproj1 in enumerate(dithers_reproj1):

        # Get out coordinates where data is valid, so we can do a linear
        # extent test
        ii, jj = np.indices(dither_reproj1.array.shape, dtype=float)
        ii[np.where(np.isnan(dither_reproj1.array))] = np.nan
        jj[np.where(np.isnan(dither_reproj1.array))] = np.nan

        dither1_iaxis = np.nanmax(ii) - np.nanmin(ii)
        dither1_jaxis = np.nanmax(jj) - np.nanmin(jj)

        for dither_idx2, dither_reproj2 in enumerate(dithers_reproj2):

            if dither_reproj2.overlaps(dither_reproj1):
                # Get diffs, remove NaNs
                diff = dither_reproj2 - dither_reproj1
                diff_arr = diff.array
                diff_foot = diff.footprint
                diff_arr[diff == 0] = np.nan
                diff_arr[diff_foot == 0] = np.nan

                # Get out coordinates where data is valid, so we can do a linear
                # extent test
                ii, jj = np.indices(dither_reproj2.array.shape, dtype=float)
                ii[np.where(np.isnan(dither_reproj2.array))] = np.nan
                jj[np.where(np.isnan(dither_reproj2.array))] = np.nan

                dither2_iaxis = np.nanmax(ii) - np.nanmin(ii)
                dither2_jaxis = np.nanmax(jj) - np.nanmin(jj)

                ii, jj = np.indices(diff_arr.shape, dtype=float)
                ii[np.where(np.isnan(diff_arr))] = np.nan
                jj[np.where(np.isnan(diff_arr))] = np.nan

                # If the slices are all NaNs, then we can just move on
                if np.all(np.isnan(ii)):
                    continue
                diff_iaxis = np.nanmax(ii) - np.nanmin(ii)
                diff_jaxis = np.nanmax(jj) - np.nanmin(jj)

                # Include everything, but flag if we've hit the minimum linear extent
                if diff_iaxis > dither1_iaxis * min_linear_frac \
                        or diff_jaxis > dither1_jaxis * min_linear_frac \
                        or diff_iaxis > dither2_iaxis * min_linear_frac \
                        or diff_jaxis > dither2_jaxis * min_linear_frac:
                    lin_size = 1

                diff = diff_arr[np.isfinite(diff_arr)].tolist()
                n_pix += len(diff)

                diffs.extend(diff)

                if plot_name is not None:
                    if len(diff) > 0:
                        vmin, vmax = np.nanpercentile(diff, [1, 99])
                        axs[dither_idx1, dither_idx2].imshow(diff_arr,
                                                             origin='lower',
                                                             vmin=vmin,
                                                             vmax=vmax,
                                                             interpolation='nearest',
                                                             )
            if plot_name is not None:
                axs[dither_idx1, dither_idx2].set_axis_off()

    if plot_name is not None:
        if n_pix > 0:
            plt.savefig('%s_ims.png' % plot_name, bbox_inches='tight')
            plt.savefig('%s_ims.pdf' % plot_name, bbox_inches='tight')
        plt.close()

    if n_pix > 0:

        # Sigma-clip to remove outliers in the distribution
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, delta, rms = sigma_clipped_stats(diffs, maxiters=maxiters)

        if plot_name is not None:
            # Get histogram range
            diffs_hist = None

            if max_points is not None:
                if len(diffs) > max_points:
                    diffs_hist = random.sample(diffs, max_points)
            if diffs_hist is None:
                diffs_hist = copy.deepcopy(diffs)

            hist_range = np.nanpercentile(diffs_hist, [0.01, 99.99])

            plt.figure(figsize=(5, 4))
            plt.hist(diffs_hist,
                     histtype='step',
                     bins=20,
                     range=hist_range,
                     color='gray',
                     )
            plt.axvline(delta,
                        c='k',
                        ls='--',
                        )

            plt.xlabel('Diff (MJy/sr)')
            plt.ylabel('N')

            plt.tight_layout()

            plt.savefig('%s_hist.pdf' % plot_name, bbox_inches='tight')
            plt.savefig('%s_hist.png' % plot_name, bbox_inches='tight')
            plt.close()

    else:
        delta = None
        rms = 0

    gc.collect()

    return n_pix, delta, rms, lin_size


def reproject_dither(file,
                     optimal_wcs,
                     optimal_shape,
                     hdu_type='data',
                     do_sigma_clip=False,
                     stacked_image=False,
                     ):
    """Reproject an image to an optimal WCS"""

    dq_bits = interpret_bit_flags('~DO_NOT_USE+NON_SCIENCE', flag_name_map=pixel)

    if not stacked_image:
        with datamodels.open(file) as hdu:

            if hdu_type == 'data':
                data = copy.deepcopy(hdu.data)
            elif hdu_type == 'var_rnoise':
                data = copy.deepcopy(hdu.var_rnoise)
            else:
                raise Warning('Unsure how to deal with hdu_type %s' % hdu_type)

            dq_bit_mask = bitfield_to_boolean_mask(
                hdu.dq.astype(np.uint8),
                dq_bits,
                good_mask_value=0,
                dtype=np.uint8
            )

            wcs = hdu.meta.wcs.to_fits_sip()
            w_in = WCS(wcs)
    else:
        with fits.open(file) as hdu:
            data = copy.deepcopy(hdu['SCI'].data)
            wcs = hdu['SCI'].header
            w_in = WCS(wcs)
        dq_bit_mask = None

    sig_mask = None
    if do_sigma_clip:
        sig_mask = make_source_mask(data,
                                    mask=dq_bit_mask,
                                    dilate_size=7,
                                    )
        sig_mask = sig_mask.astype(int)

    data[data == 0] = np.nan

    # Find the minimal shape for the reprojection. This is from the astropy reproject routines
    ny, nx = data.shape
    xc = np.array([-0.5, nx - 0.5, nx - 0.5, -0.5])
    yc = np.array([-0.5, -0.5, ny - 0.5, ny - 0.5])
    xc_out, yc_out = optimal_wcs.world_to_pixel(w_in.pixel_to_world(xc, yc))

    if np.any(np.isnan(xc_out)) or np.any(np.isnan(yc_out)):
        imin = 0
        imax = optimal_shape[1]
        jmin = 0
        jmax = optimal_shape[0]
    else:
        imin = max(0, int(np.floor(xc_out.min() + 0.5)))
        imax = min(optimal_shape[1], int(np.ceil(xc_out.max() + 0.5)))
        jmin = max(0, int(np.floor(yc_out.min() + 0.5)))
        jmax = min(optimal_shape[0], int(np.ceil(yc_out.max() + 0.5)))

    if imax < imin or jmax < jmin:
        return

    wcs_out_indiv = optimal_wcs[jmin:jmax, imin:imax]
    shape_out_indiv = (jmax - jmin, imax - imin)

    data_reproj_small = reproject_interp((data, wcs),
                                         output_projection=wcs_out_indiv,
                                         shape_out=shape_out_indiv,
                                         return_footprint=False,
                                         )

    # Mask out bad DQ, but only for unstacked images
    if not stacked_image:
        dq_reproj_small = reproject_interp((dq_bit_mask, wcs),
                                           output_projection=wcs_out_indiv,
                                           shape_out=shape_out_indiv,
                                           return_footprint=False,
                                           order='nearest-neighbor'
                                           )
        data_reproj_small[dq_reproj_small == 1] = np.nan

    if do_sigma_clip:
        sig_mask_reproj_small = reproject_interp((sig_mask, wcs),
                                                 output_projection=wcs_out_indiv,
                                                 shape_out=shape_out_indiv,
                                                 return_footprint=False,
                                                 order='nearest-neighbor'
                                                 )
        data_reproj_small[sig_mask_reproj_small == 1] = np.nan

    footprint = np.ones_like(data_reproj_small)
    footprint[np.logical_or(data_reproj_small == 0,
                            ~np.isfinite(data_reproj_small)
                            )] = 0
    data_array = ReprojectedArraySubset(data_reproj_small, footprint, imin, imax, jmin, jmax)

    del hdu
    gc.collect()

    return data_array


def parallel_reproject_weight(idx,
                              files,
                              optimal_wcs,
                              optimal_shape,
                              weight_type='exptime'
                              ):
    """Function to parallelise reprojecting with associated weights"""

    allowed_weight_types = ['exptime', 'ivm']

    file = files[idx]

    data_array = reproject_dither(file,
                                  optimal_wcs=optimal_wcs,
                                  optimal_shape=optimal_shape,
                                  )
    # Set any bad data to 0
    data_array.array[np.isnan(data_array.array)] = 0

    if weight_type == 'exptime':

        with datamodels.open(file) as model:
            # Create array of exposure time
            exptime = model.meta.exposure.exposure_time
        del model
        weight_array = copy.deepcopy(data_array)
        weight_array.array[np.isfinite(weight_array.array)] = exptime
        weight_array.array[data_array.array == 0] = 0

    elif weight_type == 'ivm':

        # Reproject the VAR_RNOISE array and take inverse
        weight_array = reproject_dither(file,
                                        optimal_wcs=optimal_wcs,
                                        optimal_shape=optimal_shape,
                                        hdu_type='var_rnoise',
                                        )
        weight_array.array = weight_array.array ** -1
        weight_array.array[np.isnan(weight_array.array)] = 0

    else:
        raise ValueError('weight_type should be one of %s' % allowed_weight_types)

    return idx, (file, data_array, weight_array)


def weighted_reproject_image(files,
                             weight_type='exptime',
                             procs=1,
                             ):
    """Get weighted mean reprojected image

    Args:
        files (list): Files to reproject
        weight_type (str): How to weight for the mean image. Options are exposure time
            (exptime) and inverse readnoise (ivm). Default 'exptime'
        procs (int): Number of processes to use. Defaults to 1.
    """

    allowed_weight_types = ['exptime', 'ivm']

    if weight_type not in allowed_weight_types:
        raise ValueError('weight_type should be one of %s' % allowed_weight_types)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        optimal_wcs, optimal_shape = find_optimal_celestial_wcs(files, hdu_in='SCI')

    data_reproj = [None] * len(files)
    weight_reproj = [None] * len(files)
    files_reproj = [None] * len(files)

    if procs == 1:
        for idx in range(len(files)):
            i, result = parallel_reproject_weight(idx,
                                                  files,
                                                  optimal_wcs=optimal_wcs,
                                                  optimal_shape=optimal_shape,
                                                  weight_type=weight_type,
                                                  )

            files_reproj[i] = result[0]
            data_reproj[i] = result[1]
            weight_reproj[i] = result[2]

            gc.collect()
    else:

        with mp.get_context('fork').Pool(procs) as pool:

            for i, result in tqdm(pool.imap_unordered(partial(parallel_reproject_weight,
                                                              files=files,
                                                              optimal_wcs=optimal_wcs,
                                                              optimal_shape=optimal_shape,
                                                              weight_type=weight_type,
                                                              ),
                                                      range(len(files))),
                                  total=len(files),
                                  desc='Creating weighted reprojects',
                                  ascii=True):
                files_reproj[i] = result[0]
                data_reproj[i] = result[1]
                weight_reproj[i] = result[2]

            pool.close()
            pool.join()
            gc.collect()

    return files_reproj, data_reproj, weight_reproj, optimal_wcs


def get_dither_match_plot_name(files1,
                               files2,
                               plot_dir,
                               ):
    """Make a plot name from list of files for dither matching"""
    if isinstance(files1, list):
        files1_name_split = os.path.split(files1[0])[-1].split('_')

        # Since these should be dither groups, blank out the dither
        files1_name_split[2] = 'XXXXX'
    else:
        files1_name_split = os.path.split(files1)[-1].split('_')
    plot_to_name = '_'.join(files1_name_split[:-1])

    if isinstance(files2, list):
        files2_name_split = os.path.split(files2[0])[-1].split('_')

        # Since these should be dither groups, blank out the dither
        files2_name_split[2] = 'XXXXX'
    else:
        files2_name_split = os.path.split(files2)[-1].split('_')
    plot_from_name = '_'.join(files2_name_split[:-1])

    plot_name = os.path.join(plot_dir,
                             '%s_to_%s' % (plot_from_name, plot_to_name),
                             )

    return plot_name


def get_dither_reproject(file,
                         optimal_wcs,
                         optimal_shape,
                         do_sigma_clip=False,
                         stacked_image=False,
                         ):
    """Reproject dithers, maintaining list structure

    Args:
        file: List or single file to reproject
        optimal_wcs: WCS to reproject to
        optimal_shape: output array shape for the WCS
        do_sigma_clip (bool): Whether to perform sigma-clipping on data or not. Defaults to False
        stacked_image (bool): Whether this is a stacked image or not. Defaults to False
    """

    if isinstance(file, list):
        dither_reproj = [reproject_dither(i, optimal_wcs=optimal_wcs,
                                          optimal_shape=optimal_shape,
                                          do_sigma_clip=do_sigma_clip,
                                          stacked_image=stacked_image,
                                          )
                         for i in file]
    else:
        dither_reproj = reproject_dither(file,
                                         optimal_wcs=optimal_wcs,
                                         optimal_shape=optimal_shape,
                                         do_sigma_clip=do_sigma_clip,
                                         stacked_image=stacked_image,
                                         )

    return dither_reproj


def parallel_get_dither_reproject(idx,
                                  files,
                                  optimal_wcs,
                                  optimal_shape,
                                  do_sigma_clip=False,
                                  stacked_image=False,
                                  ):
    """Light function to parallelise get_dither_reproject"""

    dither_reproj = get_dither_reproject(files[idx],
                                         optimal_wcs=optimal_wcs,
                                         optimal_shape=optimal_shape,
                                         do_sigma_clip=do_sigma_clip,
                                         stacked_image=stacked_image,
                                         )

    return idx, dither_reproj


def calculate_delta(files,
                    procs=None,
                    plot_dir=None,
                    max_points=10000,
                    do_sigma_clip=False,
                    stacked_image=False,
                    ):
    """Match relative offsets between tiles

    Args:
        files (list): List of files to match
        procs (int, optional): Number of processes to run in parallel. Defaults to None,
            which is series
        plot_dir (str): Where to save diagnostic plots to. Defaults to None, which will
            not save anything
        max_points (int): Maximum points to include in histogram plots. This step can
            be slow so this speeds it up. Defaults to 10000
        do_sigma_clip (bool): Whether to do sigma-clipping on data when reprojecting. Defaults to False
        stacked_image (bool): Whether this is a stacked image or not. Defaults to False
    """

    files = files

    deltas = np.zeros([len(files),
                       len(files)])
    weights = np.zeros_like(deltas)
    rmses = np.zeros_like(deltas)
    lin_sizes = np.ones_like(deltas)

    # Reproject all the HDUs. Start by building the optimal WCS
    if isinstance(files[0], list):
        files_flat = list(itertools.chain(*files))
    else:
        files_flat = copy.deepcopy(files)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        optimal_wcs, optimal_shape = find_optimal_celestial_wcs(files_flat, hdu_in='SCI')

    if procs is None:
        # Use a serial method

        # Reproject dithers, maintaining structure
        dither_reproj = []

        for file in files:
            dither_reproj.append(get_dither_reproject(file,
                                                      optimal_wcs=optimal_wcs,
                                                      optimal_shape=optimal_shape,
                                                      do_sigma_clip=do_sigma_clip,
                                                      stacked_image=stacked_image,
                                                      )
                                 )

        for i in range(len(files)):
            for j in range(i + 1, len(files)):

                plot_name = None
                if plot_dir is not None:
                    plot_name = get_dither_match_plot_name(files[i],
                                                           files[j],
                                                           plot_dir,
                                                           )

                n_pix, delta, rms, lin_size = background_match_dithers(dither_reproj[i],
                                                                       dither_reproj[j],
                                                                       plot_name=plot_name,
                                                                       max_points=max_points,
                                                                       )

                # These are symmetrical by design

                if n_pix == 0 or delta is None:
                    continue

                deltas[j, i] = delta
                weights[j, i] = n_pix
                rmses[j, i] = rms
                lin_sizes[j, i] = lin_size

                deltas[i, j] = -delta
                weights[i, j] = n_pix
                rmses[i, j] = rms
                lin_sizes[i, j] = lin_size

        gc.collect()

    else:

        # We can multiprocess this, since each calculation runs independently

        n_procs = np.nanmin([procs, len(files)])

        with mp.get_context('fork').Pool(n_procs) as pool:

            dither_reproj = list([None] * len(files))

            for i, result in tqdm(pool.imap_unordered(partial(parallel_get_dither_reproject,
                                                              files=files,
                                                              optimal_wcs=optimal_wcs,
                                                              optimal_shape=optimal_shape,
                                                              do_sigma_clip=do_sigma_clip,
                                                              stacked_image=stacked_image,
                                                              ),
                                                      range(len(files))),
                                  total=len(files),
                                  desc='Reprojecting files',
                                  ascii=True):
                dither_reproj[i] = result

            pool.close()
            pool.join()
            gc.collect()

        all_ijs = [(i, j) for i in range(len(files)) for j in range(i + 1, len(files))]

        ijs = []
        delta_vals = []
        n_pix_vals = []
        rms_vals = []
        lin_size_vals = []

        for ij in tqdm(all_ijs, ascii=True, desc='Calculating delta matrix'):
            ij, delta, n_pix, rms, lin_size = parallel_delta_matrix(ij,
                                                                    files=files,
                                                                    dither_reproj=dither_reproj,
                                                                    plot_dir=plot_dir,
                                                                    max_points=max_points,
                                                                    )

            ijs.append(ij)
            delta_vals.append(delta)
            n_pix_vals.append(n_pix)
            rms_vals.append(rms)
            lin_size_vals.append(lin_size)

        for idx, ij in enumerate(ijs):
            i = ij[0]
            j = ij[1]

            if n_pix_vals[idx] == 0 or delta_vals[idx] is None:
                continue

            deltas[j, i] = delta_vals[idx]
            weights[j, i] = n_pix_vals[idx]
            rmses[j, i] = rms_vals[idx]
            lin_sizes[j, i] = lin_size_vals[idx]

            deltas[i, j] = -delta_vals[idx]
            weights[i, j] = n_pix_vals[idx]
            rmses[i, j] = rms_vals[idx]
            lin_sizes[i, j] = lin_size_vals[idx]

        gc.collect()

    return deltas, weights, rmses, lin_sizes


def parallel_delta_matrix(ij,
                          dither_reproj,
                          files,
                          plot_dir=None,
                          max_points=None,
                          ):
    """Function to parallelise up getting delta matrix values"""

    i = ij[0]
    j = ij[1]

    plot_name = None
    if plot_dir is not None:
        plot_name = get_dither_match_plot_name(files[i],
                                               files[j],
                                               plot_dir,
                                               )

    with threadpool_limits(limits=1, user_api=None):
        n_pix, delta, rms, lin_size = background_match_dithers(dithers_reproj1=dither_reproj[i],
                                                               dithers_reproj2=dither_reproj[j],
                                                               plot_name=plot_name,
                                                               max_points=max_points,
                                                               )

    gc.collect()

    return ij, delta, n_pix, rms, lin_size


def parallel_match_dithers(dither,
                           in_band_dir,
                           step_ext,
                           plot_dir=None,
                           do_sigma_clip=False,
                           weight_method='equal',
                           ):
    """Function to parallelise up matching dithers"""

    with threadpool_limits(limits=1, user_api=None):
        dither_files = glob.glob(os.path.join(in_band_dir,
                                              '%s*_%s.fits' % (dither, step_ext))
                                 )

        dither_files.sort()

        delta_matrix, npix_matrix, rms_matrix, lin_size_matrix = calculate_delta(dither_files,
                                                                                 plot_dir=plot_dir,
                                                                                 do_sigma_clip=do_sigma_clip
                                                                                 )
        deltas = find_optimum_deltas(delta_matrix,
                                     npix_matrix,
                                     rms_matrix,
                                     lin_size_matrix,
                                     weight_method=weight_method,
                                     )

    return deltas, dither_files


def find_optimum_deltas(delta_mat,
                        npix_mat,
                        rms_mat,
                        lin_size_mat,
                        min_area_percent=0.002,
                        rms_sig_limit=2,
                        weight_method='equal',
                        ):
    """Get optimum deltas from a delta/weight matrix.

    Taken from the JWST skymatch step, with some edits to remove potentially bad fits due
    to small areal overlaps, or noisy diffs, and various weighting schemes

    Args:
        delta_mat (np.ndarray): Matrix of delta values
        npix_mat (np.ndarray): Matrix of number of pixel values for calculating delta
        rms_mat (np.ndarray): Matrix of RMS values
        lin_size_mat (np.ndarray): 1/0 array for whether overlaps pass minimum linear extent
        min_area_percent (float): Minimum percentage of average areal overlap to remove tiles.
            Defaults to 0.002 (0.2%)
        rms_sig_limit (float): Sigma limit for cutting off noisy fits. Defaults to 2
        weight_method: How to weight in least-squares minimization. Options are 'equal' (equal weighting),
            'npix' (weight by number of pixels), and 'rms' (weight by rms of the delta values).
            Defaults to 'equal'

    """

    delta_mat = copy.deepcopy(delta_mat)
    npix_mat = copy.deepcopy(npix_mat)
    rms_mat = copy.deepcopy(rms_mat)
    lin_size_mat = copy.deepcopy(lin_size_mat)

    ns = delta_mat.shape[0]

    # Remove things with weights less than min_area_percent of the average weight
    avg_npix_val = np.nanmean(npix_mat[npix_mat != 0])
    small_area_idx = np.where(np.logical_and(npix_mat < min_area_percent * avg_npix_val,
                                             delta_mat != 0)
                              )
    delta_mat[small_area_idx] = 0
    npix_mat[small_area_idx] = 0
    rms_mat[small_area_idx] = 0

    # Remove things that haven't passed the small area test
    delta_mat[lin_size_mat == 0] = 0
    npix_mat[lin_size_mat == 0] = 0
    rms_mat[lin_size_mat == 0] = 0

    # Remove fits with RMS values twice that of the mean
    avg_rms_val = np.nanmean(rms_mat[npix_mat != 0])
    sig_rms_val = np.nanstd(rms_mat[npix_mat != 0])
    rms_idx = np.where(rms_mat > avg_rms_val + rms_sig_limit * sig_rms_val)
    delta_mat[rms_idx] = 0
    npix_mat[rms_idx] = 0
    rms_mat[rms_idx] = 0

    # Make sure we've got rid of everything we need to
    delta_mat[npix_mat == 0] = 0
    delta_mat[rms_mat == 0] = 0

    # Create weight matrix
    if weight_method == 'equal':
        # Weight evenly
        weight = np.ones_like(delta_mat)
        weight[delta_mat == 0] = 0
    elif weight_method == 'npix':
        # Weight by straight number of pixels
        weight = 0.5 * (npix_mat + npix_mat.T)
    elif weight_method == 'rms':
        # Weight by inverse variance of the fit
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            weight = 0.5 * (rms_mat + rms_mat.T)
            weight = weight ** -2
            weight[~np.isfinite(weight)] = 0
    else:
        raise Warning('weight_method %s not known!' % weight_method)

    neq = 0
    for i in range(ns):
        for j in range(i + 1, ns):
            if delta_mat[i, j] != 0 and npix_mat[i, j] > 0 and weight[i, j] > 0:
                neq += 1

    # Create arrays for coefficients and free terms
    K = np.zeros((neq, ns), dtype=float)
    F = np.zeros(neq, dtype=float)
    invalid = ns * [True]

    # Process intersections between the rest of the images
    ieq = 0
    for i in range(0, ns):
        for j in range(i + 1, ns):
            # Only pull out valid intersections
            if delta_mat[i, j] != 0 and weight[i, j] > 0 and npix_mat[i, j] > 0:
                K[ieq, i] = weight[i, j]
                K[ieq, j] = -weight[i, j]

                F[ieq] = weight[i, j] * delta_mat[i, j]
                invalid[i] = False
                invalid[j] = False
                ieq += 1

    rank = np.linalg.matrix_rank(K, 1.0e-12)

    if rank < ns - 1:
        logging.warning("There are more unknown sky values ({}) to be solved for"
                        .format(ns))
        logging.warning("than there are independent equations available "
                        "(matrix rank={}).".format(rank))
        logging.warning("Sky matching (delta) values will be computed only for")
        logging.warning("a subset (or more independent subsets) of input images.")

    invK = np.linalg.pinv(K, rcond=1.0e-12)

    deltas = np.dot(invK, F)
    deltas[np.asarray(invalid, dtype=bool)] = np.nan

    return deltas


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
            with fits.open(file, memmap=False) as hdu:
                if background_name in hdu[0].header['TARGPROP'].lower():
                    f_type = 'bgr'
        else:
            raise Warning('check_type %s not known' % check_type)
    else:
        f_type = 'sci'
    with fits.open(file, memmap=False) as hdu:
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
    elif band in NIRCAM_BANDS:
        band_type = 'nircam'

        # Also pull out the distinction between short and long NIRCAM
        if int(band[1:4]) <= 212:
            short_long = 'nircam_short'
        else:
            short_long = 'nircam_long'
    else:
        raise Warning('Band type %s not known!' % band)

    pixel_scale = PIXEL_SCALES[short_long]

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


def get_kws(parameter_dict,
            func,
            band,
            target,
            ):
    """Set up kwarg dict for a function"""

    args = get_default_args(func)

    func_kws = {}
    for arg in args.keys():

        if arg in parameter_dict.keys():
            arg_val = parse_parameter_dict(parameter_dict,
                                           arg,
                                           band,
                                           target,
                                           )
            if arg_val == 'VAL_NOT_FOUND':
                arg_val = args[arg]
        else:
            arg_val = args[arg]

        func_kws[arg] = arg_val

    return func_kws


def lv3_update_fits_wcsinfo(im,
                            hdr):
    """Quick wrapper to fix up level 3 datamodel wcsinfo"""

    # update meta.wcsinfo with FITS keywords except for naxis*
    del hdr['naxis*']

    # maintain convention of lowercase keys
    hdr_dict = {k.lower(): v for k, v in hdr.items()}

    # delete naxis, cdelt, pc from wcsinfo
    rm_keys = ['naxis', 'cdelt1', 'cdelt2',
               'pc1_1', 'pc1_2', 'pc2_1', 'pc2_2',
               'a_order', 'b_order', 'ap_order', 'bp_order']

    rm_keys.extend(f"{s}_{i}_{j}" for i in range(10) for j in range(10)
                   for s in ['a', 'b', 'ap', 'bp'])

    for key in rm_keys:
        if key in im.meta.wcsinfo.instance:
            del im.meta.wcsinfo.instance[key]

    # update meta.wcs_info with fit keywords
    im.meta.wcsinfo.instance.update(hdr_dict)

    return im


def get_lv3_wcs(im):
    """Get a useful WCS from a JWST mosaic"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        fits_hdr = im.meta.wcs.to_fits()[0]
        wcs_hdr = im.meta.wcsinfo.instance
        naxis1, naxis2 = fits_hdr['NAXIS1'], fits_hdr['NAXIS2']
        wcs_hdr['naxis1'] = naxis1
        wcs_hdr['naxis2'] = naxis2

        wcs_im = WCS(wcs_hdr)

    return wcs_im


def transform_wcs_gwcs(wcs):
    """Convert WCS to gWCS"""

    jwst_hdr = wcs.to_header()
    tform = gwcs.wcs.utils.make_fitswcs_transform(jwst_hdr)
    new_gwcs = gwcs.WCS(forward_transform=tform, output_frame='world')

    return jwst_hdr, new_gwcs


def make_stacked_image(files,
                       out_name,
                       ):
    """Create a quick stacked image from a series of input images"""

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        hdus = []

        dq_bits = interpret_bit_flags('~DO_NOT_USE+NON_SCIENCE', flag_name_map=pixel)

        for file in files:
            hdu = fits.open(file)

            # Mask bad data
            dq_bit_mask = bitfield_to_boolean_mask(
                hdu['DQ'].data.astype(np.uint8),
                dq_bits,
                good_mask_value=0,
                dtype=np.uint8
            )

            hdu['SCI'].data[dq_bit_mask != 0] = np.nan

            hdus.append(hdu)

        output_projection, shape_out = find_optimal_celestial_wcs(hdus, hdu_in='SCI')
        stacked_image, stacked_foot = reproject_and_coadd(hdus,
                                                          output_projection=output_projection,
                                                          shape_out=shape_out,
                                                          hdu_in='SCI',
                                                          reproject_function=reproject_interp,
                                                          )

        hdr = output_projection.to_header()

        hdu = fits.ImageHDU(data=stacked_image, header=hdr, name='SCI')
        hdu.writeto(out_name,
                    overwrite=True,
                    )

        del hdus
        gc.collect()

    return True


def parallel_make_stack_image(dither,
                              in_dir,
                              out_dir,
                              step_ext,
                              ):
    """Parallelise up making a stacked image"""
    dither_files = glob.glob(os.path.join(in_dir,
                                          '%s*_%s.fits' % (dither, step_ext))
                             )
    dither_files.sort()

    # Create output name
    file_name_split = os.path.split(dither_files[0])[-1].split('_')
    file_name_split[2] = '*'
    out_name = '_'.join(file_name_split)
    out_name = os.path.join(out_dir, out_name)

    make_stacked_image(dither_files,
                       out_name=out_name,
                       )

    return True


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
                 psf_model_parameter_dict=None,
                 destripe_parameter_dict='phangs',
                 dither_stripe_sub_parameter_dict=None,
                 dither_match_parameter_dict=None,
                 astrometric_catalog_parameter_dict=None,
                 astrometry_parameter_dict='phangs',
                 lyot_method='mask',
                 dither_match_short_nircam_chips=True,
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
                 alignment_mapping_mode='cross_corr',
                 wcs_adjust_dict=None,
                 psf_model_dict=None,
                 correct_lv1_wcs=False,
                 crds_url='https://jwst-crds.stsci.edu',
                 webb_psf_dir=None,
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
            * psf_model_parameter_dict (dict): As `lv1_parameter_dict`, but for PSF modelling
            * destripe_parameter_dict (dict): As `lv1_parameter_dict`, but for the destriping procedure
            * dither_stripe_sub_parameter_dict (dict): As `lv1_parameter_dict`, but for the dither stripe subtraction
                procedure
            * dither_match_parameter_dict (dict): As `lv1_parameter_dict`, but for the dither matching procedure
            * astrometric_catalog_parameter_dict (dict): As `lv1_parameter_dict`, but for generating the astrometric
                catalog
            * astrometry_parameter_dict (dict): As `lv1_parameter_dict`, but for astrometric alignment
            * lyot_method (str): Method to account for mistmatch lyot coronagraph in MIRI imaging. Can either mask with
                `mask`, mask only in overlapping region with `mask_overlap',
                or adjust to main chip with `adjust`. Defaults to `mask`
            * dither_match_short_nircam_chips (bool): In dither matching, whether to do a second step where all the
                chips in a dither are matched before the final matching. Defaults to True, but should be turned off
                for dither patterns where the chips don't end up overlapping
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
            * alignment_mapping (dict): Dictionary to map basing alignments off shifts/cross-correlation with other
                aligned band. Should be of the form {'band': 'reference_band'}
            * alignment_mapping_mode (str): Whether to apply the shift from another band's alignment ('shift') or try
                to cross-correlate to match ('cross_corr'). Defaults to 'cross_corr'
            * wcs_adjust_dict (dict): dict to adjust image group WCS before tweakreg step. Should be of form
                {target: {'visit': {'matrix': [[1, 0], [0, 1]], 'shift': [dx, dy]}}}. Defaults to None.
            * psf_model_dict (dict): dict of initial guesses for saturation to replace in images. Should be of form
                {target: {band: [ [ra, dec], [ra, dec] ]}}. Defaults to None
            * correct_lv1_wcs (bool): Check WCS in uncal files, since there is a bug that some don't have this populated
                when pulled from the archive. Defaults to False
            * crds_url (str): URL to get CRDS files from. Defaults to 'https://jwst-crds.stsci.edu', which will be the
                latest versions of the files
            * webb_psf_dir (str): Directory for WebbPSF. Defaults to None, which will then not import the PSF modelling
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
        global webbpsf
        global PSFSubtraction

        import jwst
        from jwst.pipeline import calwebb_detector1, calwebb_image2, calwebb_image3
        from jwst.tweakreg import TweakRegStep
        from jwst.skymatch import SkyMatchStep
        from jwst.source_catalog import SourceCatalogStep

        self.webbpsf_imported = False
        if webb_psf_dir is not None:
            os.environ['WEBBPSF_PATH'] = webb_psf_dir
            import webbpsf
            from psf_subtraction import PSFSubtraction
            self.webbpsf_imported = True

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
        self.alignment_mapping_mode = alignment_mapping_mode

        if psf_model_parameter_dict is None:
            psf_model_parameter_dict = {}
        if psf_model_dict is None:
            psf_model_dict = {}
        self.psf_model_parameter_dict = psf_model_parameter_dict
        self.psf_model_dict = psf_model_dict

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
            }

            # Old version, using median filter
            # destripe_parameter_dict = {
            #     'quadrants': False,
            #     'destriping_method': 'median_filter',
            #     'dilate_size': 7,
            #     'median_filter_scales': [7, 31, 63, 127, 511]
            # }

        self.destripe_parameter_dict = destripe_parameter_dict

        if dither_stripe_sub_parameter_dict is None:
            dither_stripe_sub_parameter_dict = {}
        self.dither_stripe_sub_parameter_dict = dither_stripe_sub_parameter_dict

        if dither_match_parameter_dict is None:
            dither_match_parameter_dict = {}
        self.dither_match_parameter_dict = dither_match_parameter_dict

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

        if astrometric_catalog_parameter_dict is None:
            astrometric_catalog_parameter_dict = {}
        self.astrometric_catalog_parameter_dict = astrometric_catalog_parameter_dict

        self.astrometry_parameter_dict = astrometry_parameter_dict

        self.lyot_method = lyot_method

        self.dither_match_nircam_short_chips = dither_match_short_nircam_chips

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
            procs = cpu_count()

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
            'dither_stripe_sub': 'cal',
            'bg_sub': 'cal',
            'dither_match': 'cal',
            'lyot_adjust': 'cal',
            'wcs_adjust': 'cal',
            'psf_model': 'cal',
            'lv3': 'cal',
            'astrometric_catalog': 'lv3',
            'astrometric_align': 'lv3',
        }

        out_band_dir_dict = {
            'lv1': 'rate',
            'lv2': 'cal',
            'destripe': 'destripe',
            'dither_stripe_sub': 'dither_stripe_sub',
            'bg_sub': 'bg_sub',
            'dither_match': 'dither_match',
            'lyot_adjust': 'lyot_adjust',
            'wcs_adjust': 'wcs_adjust',
            'psf_model': 'psf_model',
            'lv3': 'lv3',
            'astrometric_catalog': 'lv3',
            'astrometric_align': 'lv3',
        }

        step_ext_dict = {
            'lv1': 'uncal',
            'lv2': 'rate',
            'destripe': 'cal',
            'dither_stripe_sub': 'cal',
            'bg_sub': 'cal',
            'dither_match': 'cal',
            'lyot_adjust': 'cal',
            'wcs_adjust': 'cal',
            'psf_model': 'cal',
            'lv3': 'i2d',
            'astrometric_catalog': 'i2d',
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

            if band_type == 'nircam':
                if int(band[1:4]) <= 212:
                    band_type_short_long = 'nircam_short'
                else:
                    band_type_short_long = 'nircam_long'
            else:
                band_type_short_long = 'miri'

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

                additional_step_ext = ''
                if step not in ALLOWED_STEPS:

                    # Strip off additional extensions to the step
                    step_split = step.split('_')
                    step_parsed = '_'.join(step_split[:-1])

                    if step_parsed in ALLOWED_STEPS:
                        step = copy.deepcopy(step_parsed)
                        additional_step_ext = '_' + step_split[-1]
                    else:
                        raise Warning('Step %s not recognised!' % step)

                if in_band_dir is None:
                    in_band_dir = os.path.join(base_band_dir, in_band_dir_dict[step])
                out_band_dir = os.path.join(base_band_dir, out_band_dir_dict[step] + additional_step_ext)
                step_ext = step_ext_dict[step]

                overwrite = step + additional_step_ext in overwrites

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
                                    hdu = fits.open(raw_file, memmap=False)
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

                    out_files = glob.glob(os.path.join(out_band_dir,
                                                       '*.fits')
                                          )
                    if len(out_files) == 0 or overwrite:
                        # Run lv2 asn generation
                        asn_file = self.run_asn2(directory=in_band_dir,
                                                 band=band,
                                                 parallel=True,
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

                elif step == 'dither_stripe_sub':

                    if band_type == 'nircam':

                        cal_files = glob.glob(os.path.join(in_band_dir,
                                                           '*_%s.fits' % step_ext)
                                              )
                        cal_files.sort()

                        if len(cal_files) == 0:
                            self.logger.warning('-> No files found. Skipping')
                            shutil.rmtree(base_band_dir)
                            continue

                        out_files = glob.glob(os.path.join(out_band_dir,
                                                           '*_%s.fits' % step_ext)
                                              )
                        if len(out_files) == 0:
                            self.run_dither_stripe_sub(files=cal_files,
                                                       in_dir=in_band_dir,
                                                       out_dir=out_band_dir,
                                                       step_ext=step_ext,
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
                        elif self.lyot_method == 'mask_overlap':
                            self.mask_lyot(in_files=cal_files,
                                           out_dir=out_band_dir,
                                           check_overlap=True
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
                    # Match the backgrounds for each set of dithers, and then
                    # for each tile in a mosaic (if applicable)

                    # Match the relative backgrounds between tiles in a mosaic

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

                        # Split these into dithers per-chip
                        dithers = []
                        for cal_file in cal_files:
                            cal_file_split = os.path.split(cal_file)[-1].split('_')
                            dithers.append('_'.join(cal_file_split[:2]) + '*' + cal_file_split[-2])
                        dithers = np.unique(dithers)
                        dithers.sort()

                        # First pass where we do this per-dither, per-chip. Ensure we're not wasting processes
                        procs = np.nanmin([self.procs, len(dithers)])

                        all_deltas = []
                        all_dither_files = []

                        plot_dir = os.path.join(out_band_dir, 'plots')
                        if not os.path.exists(plot_dir):
                            os.makedirs(plot_dir)

                        with mp.get_context('fork').Pool(procs) as pool:

                            dither_match_kws = get_kws(parameter_dict=self.dither_match_parameter_dict,
                                                       func=parallel_match_dithers,
                                                       band=band,
                                                       target=self.target,
                                                       )
                            del dither_match_kws['plot_dir']

                            for deltas, dither_files in tqdm(pool.imap_unordered(partial(parallel_match_dithers,
                                                                                         in_band_dir=in_band_dir,
                                                                                         step_ext=step_ext,
                                                                                         plot_dir=plot_dir,
                                                                                         **dither_match_kws,
                                                                                         ),
                                                                                 dithers),
                                                             total=len(dithers),
                                                             desc='Matching individual dithers',
                                                             ascii=True):
                                all_deltas.append(deltas)
                                all_dither_files.append(dither_files)

                            # Apply this calculated value
                            for idx in range(len(all_deltas)):
                                deltas = copy.deepcopy(all_deltas[idx])
                                dither_files = copy.deepcopy(all_dither_files[idx])

                                for i, dither_file in enumerate(dither_files):
                                    delta = copy.deepcopy(deltas[i])
                                    self.logger.info('%s, delta=%.2f' % (os.path.split(dither_file)[-1], delta))

                                    with datamodels.open(dither_file) as hdu:
                                        hdu.data -= delta
                                        hdu.save(dither_file.replace(in_band_dir, out_band_dir))
                                        hdu.close()
                                    del hdu

                            pool.close()
                            pool.join()
                            gc.collect()

                        # Do another pass where we now match between dithers
                        if len(dithers) > 1:

                            stacked_dir = os.path.join(base_band_dir,
                                                       'stacked_matched')

                            # Flush out if it already exists, and then create
                            if os.path.exists(stacked_dir):
                                shutil.rmtree(stacked_dir)
                            if not os.path.exists(stacked_dir):
                                os.makedirs(stacked_dir)

                            # We want to create stacked images for the dithers
                            with mp.get_context('fork').Pool(procs) as pool:

                                results = []

                                for result in tqdm(pool.imap_unordered(partial(parallel_make_stack_image,
                                                                               in_dir=out_band_dir,
                                                                               out_dir=stacked_dir,
                                                                               step_ext=step_ext,
                                                                               ),
                                                                       dithers),
                                                   total=len(dithers),
                                                   desc='Creating stacked dither images',
                                                   ascii=True):
                                    results.append(result)

                                pool.close()
                                pool.join()
                                gc.collect()

                            if not np.all(results):
                                self.logger.warning('Some stacked images have failed!')

                            # all_dither_files = []
                            # for dither in dithers:
                            #     dither_files = glob.glob(os.path.join(out_band_dir,
                            #                                           '%s*_%s.fits' % (dither, step_ext))
                            #                              )
                            #     dither_files.sort()
                            #     all_dither_files.append(dither_files)

                            stacked_files = glob.glob(os.path.join(stacked_dir,
                                                                   '*_%s.fits' % step_ext,
                                                                   )
                                                      )

                            self.logger.info('Matching between (stacked) mosaic tiles')

                            delta_kws = get_kws(parameter_dict=self.dither_match_parameter_dict,
                                                func=calculate_delta,
                                                band=band,
                                                target=self.target,
                                                )
                            del delta_kws['procs'], delta_kws['plot_dir'], delta_kws['stacked_image']

                            # delta_matrix, npix_matrix, rms_matrix = calculate_delta(all_dither_files,
                            #                                                         plot_dir=plot_dir,
                            #                                                         procs=self.procs,
                            #                                                         stacked_image=True,
                            #                                                         **delta_kws
                            #                                                         )

                            delta_matrix, npix_matrix, rms_matrix, lin_size_matrix = calculate_delta(stacked_files,
                                                                                                     plot_dir=plot_dir,
                                                                                                     procs=self.procs,
                                                                                                     stacked_image=True,
                                                                                                     **delta_kws
                                                                                                     )

                            delta_kws = get_kws(parameter_dict=self.dither_match_parameter_dict,
                                                func=find_optimum_deltas,
                                                band=band,
                                                target=self.target,
                                                )

                            deltas = find_optimum_deltas(delta_matrix,
                                                         npix_matrix,
                                                         rms_matrix,
                                                         lin_size_matrix,
                                                         **delta_kws,
                                                         )

                            # Subtract the per-dither delta, do in place
                            for idx, delta in enumerate(deltas):
                                stacked_file = os.path.split(stacked_files[idx])[-1]
                                dither_files = glob.glob(os.path.join(out_band_dir,
                                                                      stacked_file,
                                                                      )
                                                         )
                                dither_files.sort()

                                for dither_file in dither_files:
                                    self.logger.info('%s, delta=%.2f' % (os.path.split(dither_file)[-1], delta))

                                    with datamodels.open(dither_file) as hdu:
                                        hdu.data -= delta
                                        hdu.save(dither_file)
                                        hdu.close()
                                    del hdu

                        gc.collect()

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

                    for hdu_in_name in tqdm(cal_files, ascii=True):

                        hdu_out_name = os.path.join(out_band_dir, hdu_in_name.split(os.path.sep)[-1])

                        if not os.path.exists(hdu_out_name) or overwrite:
                            bg_sub_kws = get_kws(parameter_dict=self.bg_sub_parameter_dict,
                                                 func=background_subtract,
                                                 band=band,
                                                 target=self.target,
                                                 )

                            with fits.open(hdu_in_name, memmap=False) as hdu:
                                hdu = background_subtract(hdu,
                                                          **bg_sub_kws)
                                hdu.writeto(hdu_out_name, overwrite=True)

                elif step == 'psf_model':

                    if not self.webbpsf_imported:
                        self.logger.warning('WebbPSF has not been imported! Cannot do PSF modelling')
                        continue

                    found_sat_coords = False
                    if self.target in self.psf_model_dict.keys():
                        if band in self.psf_model_dict[self.target].keys():
                            found_sat_coords = True
                            sat_coords = self.psf_model_dict[self.target][band]

                    if not found_sat_coords:
                        continue

                    cal_files = glob.glob(os.path.join(in_band_dir,
                                                       '*_%s.fits' % step_ext)
                                          )
                    cal_files.sort()

                    if len(cal_files) == 0:
                        self.logger.warning('-> No files found. Skipping')
                        shutil.rmtree(base_band_dir)
                        continue

                    cal_files.sort()

                    if not os.path.exists(out_band_dir):
                        os.makedirs(out_band_dir)

                    self.run_psf_model(files=cal_files,
                                       out_dir=out_band_dir,
                                       sat_coords=sat_coords,
                                       target=self.target,
                                       band_type=band_type,
                                       band=band,
                                       )

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

                elif step == 'astrometric_catalog':

                    self.generate_astrometric_catalog(in_band_dir,
                                                      band,
                                                      overwrite=overwrite,
                                                      )

                elif step == 'astrometric_align':

                    if self.use_field_in_lev3 is not None:
                        in_band_dir += '_field_' + \
                                       '_'.join(np.atleast_1d(self.use_field_in_lev3).astype(str))

                    if band in self.alignment_mapping.keys():
                        self.align_wcs_to_jwst(in_band_dir,
                                               band,
                                               mode=self.alignment_mapping_mode,
                                               overwrite=overwrite,
                                               )
                    else:

                        if 'astrometric_catalog' in steps:
                            cat_suffix = 'astro_cat.fits'
                        else:
                            cat_suffix = 'cat.ecsv'

                        self.align_wcs_to_ref(in_band_dir,
                                              band,
                                              cat_suffix=cat_suffix,
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

        # Ensure we're not wasting processes
        procs = np.nanmin([self.procs, len(files)])

        with mp.get_context('fork').Pool(procs) as pool:

            results = []

            for result in tqdm(pool.imap_unordered(partial(parallel_destripe,
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

            pool.close()
            pool.join()
            gc.collect()

    def run_dither_stripe_sub(self,
                              files,
                              in_dir,
                              out_dir,
                              step_ext,
                              band,
                              group_by_dither=False,
                              default_weight_type='exptime'
                              ):
        """Run destriping, by comparing each file to overlapping dithers

        Args:
            * files (list): List of files to loop over
            * in_dir (str): Input directory
            * out_dir (str): Where to save destriped files to
            * step_ext (str): Extension for the step (e.g. cal)
            * band (str): JWST band
            * group_by_dither (bool): Whether to group by dither (True) or do
                all files at once (False). Defaults to False
            * default_weight_type (str): Default method for weighting average
                image, if not set in config. Defaults to 'exptime'
        """

        plot_dir = os.path.join(out_dir, 'plots')

        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        dither_stripe_kws = get_kws(parameter_dict=self.dither_stripe_sub_parameter_dict,
                                    func=dither_stripe_sub,
                                    band=band,
                                    target=self.target)

        try:
            weight_type = parse_parameter_dict(self.dither_stripe_sub_parameter_dict,
                                               'weight_type',
                                               band=band,
                                               target=self.target,
                                               )
        except KeyError:
            weight_type = copy.deepcopy(default_weight_type)

        if group_by_dither:

            # Split these into dithers per-chip
            dithers = []
            for file in files:
                file_split = os.path.split(file)[-1].split('_')
                dithers.append('_'.join(file_split[:2]) + '*' + file_split[-2])
            dithers = np.unique(dithers)
            dithers.sort()

            # Ensure we're not wasting processes
            procs = np.nanmin([self.procs, len(dithers)])

            with mp.get_context('fork').Pool(procs) as pool:

                results = []

                for result in tqdm(pool.imap_unordered(partial(parallel_per_dither_stripe_sub,
                                                               in_dir=in_dir,
                                                               out_dir=out_dir,
                                                               plot_dir=plot_dir,
                                                               step_ext=step_ext,
                                                               weight_type=weight_type,
                                                               **dither_stripe_kws
                                                               ),
                                                       dithers),
                                   total=len(dithers),
                                   ascii=True,
                                   desc='Dither stripe subtraction',
                                   ):
                    results.append(result)

                pool.close()
                pool.join()
                gc.collect()

                for dither_result in results:
                    for result in dither_result:
                        if result is not None:
                            in_file = result[0]
                            stripes = result[1]
                            out_file = in_file.replace(in_dir, out_dir)

                            with fits.open(in_file, memmap=False) as hdu:
                                hdu['SCI'].data -= stripes
                                hdu.writeto(out_file, overwrite=True)
                                hdu.close()
                            del hdu
                gc.collect()

        else:

            # Ensure we're not wasting processes
            procs = np.nanmin([self.procs, len(files)])

            # Global up some variables, to speed up multiprocessing
            global files_reproj
            global data_reproj
            global weight_reproj

            files_reproj, data_reproj, weight_reproj, optimal_wcs = weighted_reproject_image(files,
                                                                                             weight_type=weight_type,
                                                                                             procs=procs,
                                                                                             )

            with mp.get_context('fork').Pool(procs) as pool:

                results = [None] * len(files_reproj)

                for i, result in tqdm(pool.imap_unordered(partial(parallel_dither_stripe_sub,
                                                                  optimal_wcs=optimal_wcs,
                                                                  out_dir=out_dir,
                                                                  plot_dir=plot_dir,
                                                                  **dither_stripe_kws
                                                                  ),
                                                          range(len(files_reproj))),
                                      total=len(files_reproj),
                                      ascii=True,
                                      desc='Dither stripe subtraction',
                                      ):
                    results[i] = result

                pool.close()
                pool.join()
                gc.collect()

            for result in results:
                if result is not None:
                    in_file = result[0]
                    stripes = result[1]
                    out_file = in_file.replace(in_dir, out_dir)

                    with fits.open(in_file, memmap=False) as hdu:
                        hdu['SCI'].data -= stripes
                        hdu.writeto(out_file, overwrite=True)
                        hdu.close()
                    del hdu
            gc.collect()

    def run_psf_model(self,
                      files,
                      out_dir,
                      sat_coords,
                      target,
                      band_type,
                      band,
                      ):
        """Run PSF modelling, looping over calibrated files

        Args:
            * files (list): List of files to loop over
            * out_dir (str): Where to save PSF modelled files to
            * sat_coords (list): [ra, dec] list of coords to fit PSFs to
            * target (str): Target to match parameter dict to
            * band_type (str): Either miri or nircam
            * band (str): JWST band
        """

        fit_dir = os.path.join(out_dir, 'fit')
        plot_dir = os.path.join(out_dir, 'plot')

        if not os.path.exists(fit_dir):
            os.makedirs(fit_dir)
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)

        for hdu_name in tqdm(files, ascii=True):

            out_file = os.path.join(out_dir,
                                    os.path.split(hdu_name)[-1],
                                    )
            if os.path.exists(out_file):
                continue

            psf_sub = PSFSubtraction(hdu_in_name=hdu_name,
                                     hdu_out_name=out_file,
                                     sat_coords=sat_coords,
                                     instrument=band_type,
                                     band=band,
                                     fit_dir=fit_dir,
                                     plot_dir=plot_dir,
                                     )

            for key in self.psf_model_parameter_dict.keys():

                value = parse_parameter_dict(self.psf_model_parameter_dict,
                                             key,
                                             band,
                                             target,
                                             )
                if value == 'VAL_NOT_FOUND':
                    continue

                recursive_setattr(psf_sub, key, value)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                psf_sub.run_fitting()

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
            with fits.open(hdu_name, memmap=False) as hdu:

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

    def mask_lyot(self,
                  in_files,
                  out_dir,
                  check_overlap=False
                  ):
        """Mask lyot coronagraph by editing DQ values

        Args:
            * in_files (list): List of files to loop over
            * out_dir (str): Where to save files to
            * check_overlap (bool): if True, then mask lyot only in the regions overlaping with anything else

        """

        lyot_i = slice(735, None)
        lyot_j = slice(None, 290)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if check_overlap:
            # lyot_i = slice(750, None)
            # lyot_j = slice(None, 280)

            to_check = []
            for hdu_name in in_files:
                with fits.open(hdu_name, memmap=False) as hdu:
                    to_check.append((WCS(hdu['SCI'].header), hdu['SCI'].data, hdu['DQ'].data, os.path.basename(hdu_name)))
        n_mask = 0
        n_lyot = 0
        for hdu_name in tqdm(in_files, ascii=True):
            with fits.open(hdu_name, memmap=False) as hdu:

                out_name = os.path.join(out_dir,
                                        os.path.split(hdu_name)[-1])

                if os.path.exists(out_name):
                    return True

                if check_overlap:

                    cur_shape = hdu['SCI'].data.shape
                    cur_wcs = WCS(hdu['SCI'].header)
                    cur_lyot = np.zeros(shape=cur_shape, dtype=bool)
                    cur_lyot[lyot_i, lyot_j] = True
                    cumulative_mask = np.zeros(shape=cur_shape, dtype=bool)

                    for cur_wcs_check in to_check:
                        if cur_wcs_check[-1] == os.path.basename(hdu_name):
                            continue
                        fake_data = cur_wcs_check[2].copy()
                        fake_data[(cur_wcs_check[2] != 0) | ~np.isfinite(cur_wcs_check[1]) |
                                  (cur_wcs_check[1] == 0)] = 100
                        fake_data[lyot_i, lyot_j] = 100
                        rd = reproject_interp((fake_data.astype(float), cur_wcs_check[0]), cur_wcs, cur_shape,
                                              return_footprint=False)
                        cumulative_mask[cur_lyot & (rd == 0)] = True
                    hdu['SCI'].data[cumulative_mask] = np.nan
                    hdu['ERR'].data[cumulative_mask] = np.nan
                    hdu['DQ'].data[cumulative_mask] = 513
                    n_mask += np.sum(cumulative_mask)
                    n_lyot += np.sum(cur_lyot)
                else:
                    hdu['SCI'].data[lyot_i, lyot_j] = np.nan
                    hdu['ERR'].data[lyot_i, lyot_j] = np.nan
                    hdu['DQ'].data[lyot_i, lyot_j] = 513  # Masks the coronagraph area like the other coronagraphs
                hdu.writeto(out_name, overwrite=True)

        if check_overlap:
            logging.info(f"{n_mask}, {n_lyot}, "
                         f"{np.round(n_mask/n_lyot*100., 2)}% pixels in the lyot area have been masked")

    def parallel_wcs_adjust(self,
                            input_file,
                            output_dir,
                            ):
        """Function for parallelising WCS adjustments

        """

        output_file = os.path.join(output_dir,
                                   os.path.split(input_file)[-1],
                                   )

        if os.path.exists(output_file):
            return True

        # Set up the WCSCorrector per tweakreg
        input_im = datamodels.open(input_file)

        ref_wcs = input_im.meta.wcs
        ref_wcsinfo = input_im.meta.wcsinfo.instance

        im = JWSTWCSCorrector(
            ref_wcs,
            ref_wcsinfo
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
            self.logger.info('No WCS adjust info found for %s. Defaulting to no shift' % visit)
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

        del input_im
        del im
        gc.collect()

        return True

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

        procs = np.nanmin([self.procs, len(input_files)])

        with mp.get_context('fork').Pool(procs) as pool:
            results = []

            for result in tqdm(pool.imap_unordered(partial(self.parallel_wcs_adjust,
                                                           output_dir=output_dir,
                                                           ),
                                                   input_files),
                               total=len(input_files),
                               ascii=True,
                               desc='wcs adjust',
                               ):
                results.append(result)

            pool.close()
            pool.join()
            gc.collect()

    def run_asn2(self,
                 directory=None,
                 band=None,
                 parallel=True,
                 process_bgr_like_science=False,
                 overwrite=False,
                 ):
        """Setup asn lv2 files

        Args:
            * directory (str): Directory for files and asn file
            * band (str): JWST filter
            * parallel (bool): Whether to write out asn file for each science target, for parallelisability.
                Defaults to True
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

        # Loop over science first, then backgrounds
        sci_tab = tab[tab['Type'] == 'sci']
        bgr_tab = tab[tab['Type'] == 'bgr']

        json_content_orig = {"asn_type": "image2",
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

        if parallel:

            asn_lv2_filename = []

            for row_id, sci_row in enumerate(sci_tab):

                asn_filename = 'asn_lv2_%s_%d.json' % (band, row_id)
                asn_lv2_filename.append(asn_filename)

                if not os.path.exists(asn_filename) or overwrite:

                    json_content = copy.deepcopy(json_content_orig)

                    json_content['products'].append({
                        'name': os.path.split(sci_row['File'])[1].split('_rate.fits')[0],
                        'members': [
                            {'expname': sci_row['File'],
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

                    with open(asn_filename, 'w') as f:
                        json.dump(json_content, f)

            # If we're processing the backgrounds like science, do that here too

            if band_type in self.bgr_observation_types and process_bgr_like_science:
                for row_id, sci_row in enumerate(bgr_tab):

                    asn_filename = 'asn_lv2_%s_bgr_%d.json' % (band, row_id)
                    asn_lv2_filename.append(asn_filename)

                    if not os.path.exists(asn_filename) or overwrite:

                        json_content = copy.deepcopy(json_content_orig)

                        json_content['products'].append({
                            'name': f'offset_{band}_{row_id + 1}',
                            'members': [
                                {'expname': sci_row['File'],
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

                        with open(asn_filename, 'w') as f:
                            json.dump(json_content, f)

        else:

            asn_lv2_filename = 'asn_lv2_%s.json' % band

            if not os.path.exists(asn_lv2_filename) or overwrite:

                json_content = copy.deepcopy(json_content_orig)

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

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if self.correct_lv1_wcs:
                    if 'MAST_API_TOKEN' not in os.environ.keys():
                        os.environ['MAST_API_TOKEN'] = input('Input MAST API token: ')

                # For speed, we want to parallelise these up by dither since we use the
                # persistence file

                dithers = np.unique(['_'.join(os.path.split(uncal_file)[-1].split('_')[:2])
                                     for uncal_file in uncal_files])

                # Ensure we're not wasting processes
                procs = np.nanmin([self.procs, len(dithers)])

                with mp.get_context('fork').Pool(procs) as pool:

                    results = []

                    for result in pool.imap_unordered(partial(self.parallel_lv1,
                                                              band=band,
                                                              output_dir=output_dir,
                                                              ),
                                                      dithers):
                        results.append(result)

                    pool.close()
                    pool.join()
                    gc.collect()

        elif pipeline_stage == 'lv2':

            if overwrite:
                os.system('rm -rf %s' % output_dir)

            if len(glob.glob(os.path.join(output_dir, '*.fits'))) == 0 or overwrite:

                os.system('rm -rf %s' % output_dir)

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                if isinstance(asn_file, str):
                    asn_file = [asn_file]

                # Ensure we're not wasting processes
                procs = np.nanmin([self.procs, len(asn_file)])

                with mp.get_context('fork').Pool(procs) as pool:

                    results = []

                    for result in pool.imap_unordered(partial(self.parallel_lv2,
                                                              band=band,
                                                              output_dir=output_dir,
                                                              ),
                                                      asn_file):
                        results.append(result)

                    pool.close()
                    pool.join()
                    gc.collect()

        elif pipeline_stage == 'lv3':

            if overwrite:
                os.system('rm -rf %s' % output_dir)

            output_fits = '%s_%s_lv3_%s_i2d.fits' % (self.target.lower(), band_type, band.lower())
            output_file = os.path.join(output_dir, output_fits)

            if not os.path.exists(output_file) or overwrite:

                os.system('rm -rf %s' % output_dir)

                if not os.path.exists(output_dir):
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

                del tweakreg
                gc.collect()

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

                del skymatch
                gc.collect()

                # Degroup again to avoid potential weirdness later
                if band_type_short_long in self.group_skymatch_dithers:
                    for i, model in enumerate(asn_file._models):
                        model.meta.observation.exposure_number = str(i)

                im3.skymatch.skip = True

                # Run the rest of the level 3 pipeline
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    im3.run(asn_file)

                del im3
                del asn_file
                gc.collect()

        else:

            raise Warning('Pipeline stage %s not recognised!' % pipeline_stage)

        os.chdir(orig_dir)

    def parallel_lv1(self,
                     group,
                     band=None,
                     output_dir=None,
                     ):
        """Function to parallelise lv1 reprocessing"""

        if band in NIRCAM_BANDS:
            band_type = 'nircam'
        elif band in MIRI_BANDS:
            band_type = 'miri'
        else:
            raise Warning('Band %s not recognised!' % band)

        uncal_files = glob.glob('%s*_uncal.fits' % group
                                )

        uncal_files.sort()

        for uncal_file in uncal_files:

            # Sometimes the WCS is catastrophically wrong. Try to correct that here
            if self.correct_lv1_wcs:
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

            detector1 = attribute_setter(detector1,
                                         parameter_dict=self.lv1_parameter_dict,
                                         band=band,
                                         target=self.target)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                # Run the level 1 pipeline
                detector1.run(uncal_file)

            del detector1
            gc.collect()

        return True

    def parallel_lv2(self,
                     asn_file,
                     band=None,
                     output_dir=None,
                     ):
        """Function to parallelise running lv2 processing"""

        config = calwebb_image2.Image2Pipeline.get_config_from_reference(asn_file)
        im2 = calwebb_image2.Image2Pipeline.from_config_section(config)
        im2.output_dir = output_dir

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

        del im2
        gc.collect()

        return True

    def generate_astrometric_catalog(self,
                                     input_dir,
                                     band,
                                     overwrite=False
                                     ):
        """Generate a catalog for absolute astrometric alignment

        Args:
            input_dir (str): Directory to search for files
            band (str): JWST band in question
            overwrite (bool): Overwrite or not. Defaults to False
        """

        jwst_files = glob.glob(os.path.join(input_dir,
                                            '*i2d.fits'))

        for jwst_file in jwst_files:

            cat_name = jwst_file.replace('i2d.fits', 'astro_cat.fits')
            if not os.path.exists(cat_name) or overwrite:

                hdu = fits.open(jwst_file, memmap=False)
                data_hdu = hdu['SCI']
                w = WCS(data_hdu)
                data = data_hdu.data

                snr = self.astrometric_catalog_parameter_dict['snr']

                mask = data == 0
                mean, median, rms = sigma_clip(data, dq_mask=mask)
                threshold = median + snr * rms

                kernel_fwhm = FWHM_PIX[band]

                daofind = DAOStarFinder(fwhm=kernel_fwhm,
                                        threshold=threshold,
                                        )

                for astro_key in self.astrometric_catalog_parameter_dict:
                    value = parse_parameter_dict(self.astrometric_catalog_parameter_dict,
                                                 astro_key,
                                                 band,
                                                 self.target,
                                                 )

                    if value == 'VAL_NOT_FOUND':
                        continue

                    recursive_setattr(daofind, astro_key, value)

                sources = daofind(data, mask=mask)

                # Add in RA and Dec
                ra, dec = w.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
                sky_coords = SkyCoord(ra * u.deg, dec * u.deg)
                sources.add_column(sky_coords, name='sky_centroid')
                sources.write(cat_name, overwrite=True)

    def align_wcs_to_ref(self,
                         input_dir,
                         band,
                         cat_suffix='cat.ecsv',
                         overwrite=False,
                         ):
        """Align JWST image to external references. Either a table or an image

        Args:
            * input_dir (str): Directory to find files to align
            * band (str): JWST band to align
            * cat_suffix (str): Suffix for the astrometric catalog. Defaults to 'cat.ecsv'
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

            ref_hdu = fits.open(self.astrometric_alignment_image, memmap=False)

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

                # Read in the source catalogue from the pipeline

                source_cat_name = jwst_file.replace('i2d.fits', cat_suffix)

                if cat_suffix.split('.')[-1] == 'ecsv':

                    sources = Table.read(source_cat_name, format='ascii.ecsv')
                    # convenience for CARTA viewing.
                    sources.write(source_cat_name.replace('.ecsv', '.fits'), overwrite=True)
                else:
                    sources = Table.read(source_cat_name)

                # Filter out extended sources
                if 'is_extended' in sources.colnames:
                    sources = sources[~sources['is_extended']]
                # Filter based on roundness and sharpness
                # sources = sources[np.logical_and(sources['sharpness'] >= 0.2,
                #                                  sources['sharpness'] <= 1.0)]
                # sources = sources[np.logical_and(sources['roundness'] >= -1.0,
                #                                  sources['roundness'] <= 1.0)]

                # Load in the datamodel, and pull in WCS to correct
                jwst_im = datamodels.open(jwst_file)
                wcs_jwst = get_lv3_wcs(jwst_im)
                wcs_jwst_corrector = FITSWCSCorrector(wcs_jwst)

                # Make a copy since we'll be overwriting this along the way
                wcs_jwst_corrector_orig = copy.deepcopy(wcs_jwst_corrector)

                # Parse down the table and convert appropriately
                jwst_tab = Table()

                # Get TPx/y out -- do everything in pixel space
                jwst_tab['TPx'], jwst_tab['TPy'] = wcs_jwst_corrector.world_to_det(sources['sky_centroid'].ra,
                                                                                   sources['sky_centroid'].dec,
                                                                                   )
                ref_tab['TPx'], ref_tab['TPy'] = wcs_jwst_corrector.world_to_det(ref_tab['RA'],
                                                                                 ref_tab['DEC'])

                # We'll also need x and y for later
                jwst_tab['x'] = sources['xcentroid']
                jwst_tab['y'] = sources['ycentroid']

                jwst_tab['ra'] = sources['sky_centroid'].ra.value
                jwst_tab['dec'] = sources['sky_centroid'].dec.value

                # Do the fit -- potentially take an iterative approach, using
                # multiple homing-in iterations
                multiple_iterations = False
                n_iterations = 0
                for key in self.astrometry_parameter_dict.keys():
                    if 'iteration' in key:
                        multiple_iterations = True
                        n_iterations += 1

                if not multiple_iterations:
                    n_iterations = 1

                wcs_aligned_fit = None

                xoffset = 0
                yoffset = 0

                for iteration in range(n_iterations):

                    # Make sure we're not overwriting WCS
                    wcs_jwst_corrector = copy.deepcopy(wcs_jwst_corrector_orig)

                    if not multiple_iterations:
                        astrometry_parameter_dict = copy.deepcopy(
                            self.astrometry_parameter_dict
                        )
                    else:
                        astrometry_parameter_dict = copy.deepcopy(
                            self.astrometry_parameter_dict['iteration%d' % (iteration + 1)]
                        )

                    # Run a match
                    match = XYXYMatch(xoffset=xoffset,
                                      yoffset=yoffset,
                                      )
                    for key in astrometry_parameter_dict.keys():

                        value = parse_parameter_dict(astrometry_parameter_dict,
                                                     key,
                                                     band,
                                                     self.target,
                                                     )
                        if value == 'VAL_NOT_FOUND':
                            continue

                        recursive_setattr(match, key, value, protected=True)

                    ref_idx, jwst_idx = match(ref_tab, jwst_tab, tp_units='pix')

                    fit_wcs_args = get_default_args(fit_wcs)

                    fit_wcs_kws = {}
                    for fit_wcs_arg in fit_wcs_args.keys():

                        if fit_wcs_arg in astrometry_parameter_dict.keys():
                            arg_val = parse_parameter_dict(astrometry_parameter_dict,
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

                    # Pull out offsets, remember there's a negative here to the shift
                    xoffset, yoffset = -wcs_aligned_fit.meta['fit_info']['shift']

                shift = wcs_aligned_fit.meta['fit_info']['shift']
                matrix = wcs_aligned_fit.meta['fit_info']['matrix']

                wcs_jwst_corrected = copy.deepcopy(wcs_jwst_corrector_orig)

                # Put the correction in and properly update header.
                wcs_jwst_corrected.set_correction(shift=shift,
                                                  matrix=matrix,
                                                  ref_tpwcs=wcs_jwst_corrector_orig,
                                                  )

                jwst_hdr, new_gwcs = transform_wcs_gwcs(wcs_jwst_corrected.wcs)
                jwst_im.meta.wcs = new_gwcs

                # Add in shift metadata
                jwst_im.meta.abs_astro_alignment = {'shift': shift,
                                                    'matrix': matrix,
                                                    }

                # Update WCS info
                jwst_im = lv3_update_fits_wcsinfo(im=jwst_im, hdr=jwst_hdr)

                jwst_im.write(aligned_file)

                # Also apply this to each individual crf file
                crf_files = glob.glob(os.path.join(input_dir,
                                                   '*_crf.fits')
                                      )

                crf_files.sort()

                # Move found CRF files, if they exist
                if len(crf_files) > 0:

                    # Ensure we're not wasting processes
                    procs = np.nanmin([self.procs, len(crf_files)])

                    with mp.get_context('fork').Pool(procs) as pool:

                        results = []

                        for result in tqdm(pool.imap_unordered(partial(parallel_tweakback,
                                                                       matrix=matrix,
                                                                       shift=shift,
                                                                       ref_tpwcs=wcs_jwst_corrector_orig,
                                                                       ),
                                                               crf_files),
                                           total=len(crf_files),
                                           ascii=True,
                                           desc='tweakback',
                                           ):
                            results.append(result)

                        pool.close()
                        pool.join()
                        gc.collect()

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

    def align_wcs_to_jwst(self,
                          input_dir,
                          band,
                          mode='cross_corr',
                          overwrite=False,
                          ):
        """Internally align image to already aligned JWST one, either through pulling out corrector terms or via
        cross-correlation

        Args:
            * input_dir (str): Directory to find files to align
            * band (str): JWST band to align
            * mode (str): Either 'cross_corr' (use cross-correlation) or 'shift' (pull out transformation from previous
                alignment). Defaults to 'cross_corr'
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

                jwst_im = datamodels.open(jwst_file)
                wcs_jwst = get_lv3_wcs(jwst_im)
                wcs_jwst_corrector = FITSWCSCorrector(wcs_jwst)
                wcs_jwst_corrector_orig = copy.deepcopy(wcs_jwst_corrector)

                if mode == 'cross_corr':
                    # ref_hdu = fits.open(ref_hdu_name, memmap=False)
                    ref_im = datamodels.open(jwst_file)
                    ref_wcs = get_lv3_wcs(jwst_im)

                    ref_data = copy.deepcopy(ref_im.data)
                    jwst_data = copy.deepcopy(jwst_im.data)

                    ref_err = copy.deepcopy(ref_im.err)
                    jwst_err = copy.deepcopy(jwst_im.err)

                    ref_data[ref_data == 0] = np.nan
                    jwst_data[jwst_data == 0] = np.nan

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        ref_data = reproject_interp((ref_data, ref_wcs),
                                                    wcs_jwst,
                                                    shape_out=jwst_data.shape,
                                                    return_footprint=False,
                                                    )

                        ref_err = reproject_interp((ref_err, ref_wcs),
                                                   wcs_jwst,
                                                   shape_out=jwst_data.shape,
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
                    shift = [-x_off, -y_off]
                    matrix = None

                    self.logger.info('Found offset of %s' % shift)

                    # Apply correction
                    wcs_jwst_corrector.set_correction(shift=shift,
                                                      ref_tpwcs=wcs_jwst_corrector_orig,
                                                      )

                    jwst_hdr, new_gwcs = transform_wcs_gwcs(wcs_jwst_corrector.wcs)
                    jwst_im.meta.wcs = new_gwcs

                    # Add in shift metadata
                    jwst_im.meta.abs_astro_alignment = {'shift': shift,
                                                        'matrix': matrix,
                                                        }

                    # Update WCS info
                    jwst_im = lv3_update_fits_wcsinfo(im=jwst_im, hdr=jwst_hdr)

                    jwst_im.write(aligned_file)

                elif mode == 'shift':

                    ref_hdu = datamodels.open(ref_hdu_name)

                    shift = ref_hdu.meta.abs_astro_alignment.shift
                    matrix = ref_hdu.meta.abs_astro_alignment.matrix

                    # Cast these to numpy, so they can be pickled properly later
                    shift = shift.astype(np.ndarray).astype(float)
                    matrix = matrix.astype(np.ndarray).astype(float)

                    wcs_jwst_corrector.set_correction(matrix=matrix,
                                                      shift=shift,
                                                      ref_tpwcs=wcs_jwst_corrector_orig,
                                                      )
                    jwst_hdr, new_gwcs = transform_wcs_gwcs(wcs_jwst_corrector.wcs)
                    jwst_im.meta.wcs = new_gwcs

                    # Update WCS info
                    jwst_im = lv3_update_fits_wcsinfo(im=jwst_im, hdr=jwst_hdr)

                    jwst_im.write(aligned_file)

                else:
                    raise ValueError('mode should be one of cross_corr, shift')

                # Also apply this to each individual crf file, if they exist

                crf_files = glob.glob(os.path.join(input_dir,
                                                   '*_crf.fits')
                                      )

                crf_files.sort()

                if len(crf_files) > 0:

                    # Ensure we're not wasting processes
                    procs = np.nanmin([self.procs, len(crf_files)])

                    with mp.get_context('fork').Pool(procs) as pool:

                        results = []

                        for result in tqdm(pool.imap_unordered(partial(parallel_tweakback,
                                                                       shift=shift,
                                                                       matrix=matrix,
                                                                       ref_tpwcs=wcs_jwst_corrector_orig,
                                                                       ),
                                                               crf_files),
                                           total=len(crf_files),
                                           ascii=True,
                                           desc='tweakback',
                                           ):
                            results.append(result)

                        pool.close()
                        pool.join()
                        gc.collect()

                    if not all(results):
                        self.logger.warning('Not all crf files tweakbacked. May cause issues!')
