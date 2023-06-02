from collections import OrderedDict
import sys, glob

import numpy as np
import scipy.stats
from scipy.odr import ODR, Model, Data, RealData

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import mad_std
from astropy.constants import iau2012 as const
from astropy.convolution import convolve_fft, Gaussian2DKernel

from scipy.interpolate import RegularGridInterpolator
from reproject import reproject_interp

import matplotlib.pyplot as plt

from scipy.stats import spearmanr, gaussian_kde
from utils_stats import *

def make_gaussian_psf(fwhm_arcsec = 1.0, oversample_by=10., outfile=None):
    """Adapted from Francesco Belfiore / Tom Williams. Could/should
    replace with radio_beam and spectral cube calculations for a full
    to/from infrastructure.

    Should add an option to force the PSF size.
    """

    pix_size_arcsec = fwhm_arcsec / oversample_by
    std_pix = fwhm_arcsec / 2.355 / pix_size_arcsec
    
    gauss_2d = Gaussian2DKernel(x_stddev=std_pix)
    
    psf_array = fits.PrimaryHDU(data=np.array(gauss_2d.array, dtype=np.float32))
    
    psf_array.header['BITPIX'] = -32
    
    psf_array.header['CRPIX1'] = (gauss_2d.array.shape[1] + 1) / 2
    psf_array.header['CRPIX2'] = (gauss_2d.array.shape[0] + 1) / 2
    
    psf_array.header['CRVAL1'] = 0.00
    psf_array.header['CRVAL2'] = 0.00
    
    psf_array.header['CDELT1'] = - pix_size_arcsec / 3600
    psf_array.header['CDELT2'] = pix_size_arcsec / 3600

    psf_array.header['CTYPE1'] = 'RA---TAN'
    psf_array.header['CTYPE2'] = 'DEC--TAN'

    if outfile is not None:
        psf_array.writeto(outfile)
        
    return(psf_array)
    
def get_pixscale(hdu):
    """From Tom Williams. Get pixel scale from header. Checks HDU header
    for pixel scale keywords, and returns a pixel scale in arcsec. If
    no suitable keyword is found, will throw up an error. From Tom
    Williams.

    Args:

        hdu (astropy.fits.PrimaryHDU): HDU to get pixel scale for.

    Returns:

        pix_scale (float): Pixel scale in arcsec.

    Raises:

        Warning: If no suitable pixel scale keyword is found in header.

    Author:

    Tom Williams

    """

    PIXEL_SCALE_NAMES = ['XPIXSIZE', 'CDELT1', 'CD1_1', 'PIXELSCL']

    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(hdu.header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ['CDELT1', 'CD1_1']:
                pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning('No pixel scale found')

def conv_with_kernel(image_hdu, kernel_hdu,
                     blank_zeros=True, set_zeros_to=np.nan,
                     allow_huge=True, preserve_nan=True, nan_treatment='interpolate',
                     outfile=None, overwrite=True):
    """Convolves in input image with an input kernel, both HDUs, and
    returns a new HDU or optionally writes to disk. From Tom Williams,
    refactored into general routine.
    """

    # Set zero values to NaN (or another user-specified value), which
    # astropy convolution handles with interpolation.
    if blank_zeros:
        image_hdu.data[image_hdu.data == 0] = set_zeros_to

    # Get the pixel scale of the image and the kernel
    
    image_pix_scale = get_pixscale(image_hdu)    
    kernel_pix_scale = get_pixscale(kernel_hdu)

    # Note the shape and grid of the kernel as input
    
    kernel_hdu_length = kernel_hdu.data.shape[0]            
    original_central_pixel = (kernel_hdu_length - 1) / 2            
    original_grid = (np.arange(kernel_hdu_length) - original_central_pixel) * kernel_pix_scale

    # Calculate kernel size after interpolating to the image pixel
    # scale. Because sometimes there's a little pixel scale rounding
    # error, subtract a little bit off the optimum size (Tom
    # Williams).
            
    interpolate_kernel_size = np.floor(kernel_hdu_length * kernel_pix_scale / image_pix_scale) - 2

    # Ensure the kernel has a central pixel
    
    if interpolate_kernel_size % 2 == 0:
        interpolate_kernel_size -= 1

    # Define a new coordinate grid onto which to project the kernel
    # but using the pixel scale of the image
    
    new_central_pixel = (interpolate_kernel_size - 1) / 2            
    new_grid = (np.arange(interpolate_kernel_size) - new_central_pixel) * image_pix_scale
    x_coords_new, y_coords_new = np.meshgrid(new_grid, new_grid)

    # Do the reprojection from the original kernel grid onto the new
    # grid with pixel scale matched to the image
    
    grid_interpolated = RegularGridInterpolator((original_grid, original_grid), kernel_hdu.data)
    kernel_interp = grid_interpolated((x_coords_new.flatten(), y_coords_new.flatten()))
    kernel_interp = kernel_interp.reshape(x_coords_new.shape)
            
    # Now with the kernel centered and matched in pixel scale to the
    # input image use the FFT convolution routine from astropy to
    # convolve.
    
    image_data_convolved = convolve_fft(image_hdu.data, kernel_interp,
                                        allow_huge=allow_huge, preserve_nan=preserve_nan,
                                        nan_treatment=nan_treatment)
    
    # Form into an HDU
    
    input_hdu_convolved = fits.PrimaryHDU(image_data_convolved, image_hdu.header)

    # If an output file name is supplied write to disk 
    
    if outfile is not None:
        input_hdu_convolved.writeto(outfile, overwrite=overwrite)

    return(input_hdu_convolved)

def align_image(hdu_to_align, target_header, hdu_in=0,
                order='bilinear', missing_value=np.nan,
                outfile=None, overwrite=True):
    """
    Aligns an image to a target header and handles reattaching the
    header to the file with updated WCS keywords.
    """

    # Run the reprojection
    reprojected_image, footprint = reproject_interp(
        hdu_to_align, target_header, hdu_in=hdu_in,
        order=order, return_footprint=True)

    # Blank missing locations outside the footprint    
    missing_mask = (footprint == 0)
    reprojected_image[missing_mask] = missing_value
    
    # Get the WCS of the target header
    target_wcs = WCS(target_header)
    target_wcs_keywords = target_wcs.to_header()
    
    # Get the WCS of the original header
    orig_header = hdu_to_align.header
    orig_wcs = WCS(orig_header)
    orig_wcs_keywords = orig_wcs.to_header()

    # Create a reprojected header using the target WCS but keeping
    # other keywords the same.
    reprojected_header = hdu_to_align.header
    for this_keyword in orig_wcs_keywords:
        if this_keyword in reprojected_header:
            del reprojected_header[this_keyword]

    for this_keyword in target_wcs_keywords:
        reprojected_header[this_keyword] = target_wcs_keywords[this_keyword]

    # Make a combined HDU merging the image and new header 
    reprojected_hdu = fits.PrimaryHDU(
        reprojected_image, reprojected_header)

    # Write or return
    if outfile is not None:
        reprojected_hdu.writeto(outfile, overwrite=overwrite)
    
    return(reprojected_hdu)

def solve_for_offset(
        comp_hdu, ref_hdu, mask_hdu=None,
        xmin=0.25, xmax=3.5, binsize=0.1,
        make_plot=True, plot_fname=None,
        label_str='Comparison'):
    """Solve for the offset between two images, optionally also allowing
    for a free slope relating them and restricting to a specific range
    of values or applying an extra spatial mask.

    Inputs:

    

    Returns:

    offset, slope : the value to subtract from COMP to match the zero
    point of REF assuming they can be related by a single scaling,
    along with the SLOPE to multiply REF by to get COMP after removing
    a DC offset. That is:

    comp = slope * ref + offset

    """

    # Can a bunch error checking, reading, and alignment stuff here
    # but for this case, assume aligned HDUs and treat the program as
    # fragile.
    
    # Identify overlap used to solve for the offset

    overlap = np.isfinite(comp_hdu.data)*np.isfinite(ref_hdu.data)

    if mask_hdu is not None:
        overlap = overlap*(mask_hdu.data)

    # Solve for the difference and note statistics
    comp_vec = comp_hdu.data[overlap]
    ref_vec = ref_hdu.data[overlap]
    diff_vec = comp_vec-ref_vec

    # Optionally make diagnostic plots
    xlo, xhi, xmid, nbin = make_spanning_bins(xmin, xmax, binsize)
    x_for_bins = ref_vec
    comp_bins = bin_data(x_for_bins,comp_vec,xlo=xlo,xhi=xhi)
    
    if make_plot:
        
        xlim_lo = -1.
        xlim_hi = 5.0

        fig, ax = plt.subplots()
                
        ax.set_xlim(xlim_lo, xlim_hi)
        ax.set_ylim(xlim_lo, xlim_hi)
        ax.grid(True, linestyle='dotted', linewidth=0.5, color='black', zorder=2)

        ax.scatter(ref_vec, comp_vec, marker='.', color='gray', s=1, zorder=1)

        xbins = comp_bins['xmid']
        ybins = (comp_bins['50'])
        lo_ybins = (comp_bins['16'])
        hi_ybins = (comp_bins['84'])

        slope, intercept, resid = iterate_ols(
            xbins, ybins, e_y=None, 
            x0=None, s2nclip=3., iters=3, guess=[0.0,1.0],
            doprint=False)
        
        ax.scatter(comp_bins['xmid'], comp_bins['50'], color='red', marker='o',s=50, zorder=5)
        ax.errorbar(xbins, ybins, [(ybins-lo_ybins), (hi_ybins-ybins)],
                     color='red', capsize=0.1, elinewidth=2, fmt='none',
                     zorder=4)

        fidx = np.arange(xlim_lo,xlim_hi,0.01)
        ax.plot(fidx, fidx*slope + intercept, linewidth=3, 
                 color='black', zorder=6, alpha=0.5, linestyle='dashed')
        
        bbox_props = dict(boxstyle="round", fc="lightgray", ec='black',alpha=0.9)
        yval = 0.95
        va = 'top'

        this_label = label_str+'\n'+'m = '+str(slope)+'\n'+'b='+str(intercept)
        ax.text(0.04, yval, this_label,
                 ha='left', va=va,
                 transform=ax.transAxes,
                 size='small', bbox=bbox_props,
                 zorder=5)
        
        if plot_fname is not None:
            plt.savefig(plot_fname, bbox_inches='tight')
        else:
            plt.show()

    # Return
        
    return(slope, intercept)

