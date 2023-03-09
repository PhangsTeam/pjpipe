# Read and convolve all of current JWST images to F2100W matched
# resolution as well as a few common round Gaussian resolutions.

import os, glob

from astropy.table import Table
import numpy as np
from astropy.io import fits

# This module holds most of the key routines

from utils_jwst import *

# -----------------------------------------------------------------------------------
# Control flow
# -----------------------------------------------------------------------------------

do_conv_to_f2100w = True
do_conv_to_gauss = True

# -----------------------------------------------------------------------------------
# Directory structure
# -----------------------------------------------------------------------------------

# This could be replaced with TOML files and markup. It just defines
# the relevant directory structure.

input_tab = Table.read('jwst_image_key.txt', format='ascii.csv',comment='#')
gal_names = np.unique(np.array(input_tab['galaxy']))
filters = np.unique(np.array(input_tab['filter']))

my_input_root = '../working_data/processed_jwst/anchored/'
my_kern_dir = '../jwst_scripts_fork/PSF/kernels/'
my_output_root = '../working_data/processed_jwst/anchored_matched_res/'

# -----------------------------------------------------------------------------------
# Loop over all galaxies and do the initial convolution 
# -----------------------------------------------------------------------------------

for this_gal in gal_names:
    
    template_filter = 'F2100W'
    template_mask = (input_tab['galaxy'] == this_gal)* \
        (input_tab['filter'] == template_filter)
    template_file = my_input_root+str(np.array(input_tab[template_mask]['filename'])[0])
    if os.path.isfile(template_file) == False:
        print("No valid template for ", template_file)
        continue
    template_header = fits.open(template_file)['SCI'].header
    
    for this_filt in filters:

        # Select the relevant table row
        tab_mask = (input_tab['galaxy'] == this_gal)*(input_tab['filter'] == this_filt)        
        input_file = my_input_root+str(np.array(input_tab[tab_mask]['filename'])[0])
        print(this_gal, this_filt, input_file)

        # Check that the input file is present
        if os.path.isfile(input_file) == False:
            print("Input file not found, skipping. ", input_file)
            continue

        # Read the input science image
        input_hdu = fits.open(input_file)['SCI']

        # Identify names of kernels
        kern_to_f2100w = my_kern_dir+this_filt+'_to_F2100W.fits'
        kern_to_gauss = my_kern_dir+this_filt+'_to_Gauss_1.150.fits'

        # ---------------------
        # Convolve to F2100W
        # ---------------------        

        if do_conv_initial_to_f2100w:
        
            output_file_name = my_output_root + this_gal + '_'+this_filt+'_atF2100W.fits'
            print("... building ", output_file_name)
            if this_filt != 'F2100W':
                kernel_hdu = fits.open(kern_to_f2100w)[0]
                convolved_hdu = conv_with_kernel(
                    input_hdu, kernel_hdu,
                    outfile=output_file_name, overwrite=True)

                aligned_hdu = align_image(
                    convolved_hdu, template_header, hdu_in=0,
                    order='bilinear', missing_value=np.nan,
                    outfile=None, overwrite=True)

                aligned_hdu.writeto(output_file_name, overwrite=True)            
            else:
                input_hdu.writeto(output_file_name, overwrite=True)
            
        # ---------------------
        # Convolve to Gaussian
        # ---------------------        

        if do_conv_initial_to_gauss:
        
            output_file_name = my_output_root + this_gal + '_'+this_filt+'_atGauss1p15.fits'
            print("... building ", output_file_name)
            kernel_hdu = fits.open(kern_to_gauss)[0]
            convolved_hdu = conv_with_kernel(
                input_hdu, kernel_hdu,
                outfile=output_file_name, overwrite=True)
            convolved_hdu.writeto(output_file_name, overwrite=True)

            # also to 4 arcsec
            output_file_name = my_output_root + this_gal + '_'+this_filt+'_atGauss4.fits'
            print("... building ", output_file_name)
            kernel_hdu = make_gaussian_psf(fwhm_arcsec = np.sqrt(4.**2-(1.15)**2), oversample_by=50., outfile=None)
            convolved_more_hdu = conv_with_kernel(
                convolved_hdu, kernel_hdu,
                outfile=output_file_name, overwrite=True)
            convolved_more_hdu.writeto(output_file_name, overwrite=True)

            # also to 7.5 arcsec
            output_file_name = my_output_root + this_gal + '_'+this_filt+'_atGauss7p5.fits'
            print("... building ", output_file_name)
            kernel_hdu = make_gaussian_psf(fwhm_arcsec = np.sqrt(7.5**2-(1.15)**2), oversample_by=200., outfile=None)
            convolved_more_hdu = conv_with_kernel(
                convolved_hdu, kernel_hdu,
                outfile=output_file_name, overwrite=True)
            convolved_more_hdu.writeto(output_file_name, overwrite=True)
            
            # and to 15 arcsec
            output_file_name = my_output_root + this_gal + '_'+this_filt+'_atGauss15.fits'
            print("... building ", output_file_name)
            kernel_hdu = make_gaussian_psf(fwhm_arcsec = np.sqrt(15.**2-(1.15)**2), oversample_by=200., outfile=None)
            convolved_more_hdu = conv_with_kernel(
                convolved_hdu, kernel_hdu,
                outfile=output_file_name, overwrite=True)
            convolved_more_hdu.writeto(output_file_name, overwrite=True)
