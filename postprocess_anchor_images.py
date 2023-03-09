# Compare convolved JWST to low resolution reference data to solve for
# overall background level of the maps

import os, glob

from astropy.table import Table
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy.interpolate import RegularGridInterpolator
from reproject import reproject_interp
import matplotlib.pyplot as plt

# This module holds most of the key routines

from utils_jwst import *

# -----------------------------------------------------------------------------------
# Control flow
# -----------------------------------------------------------------------------------

do_compare_all_bands = True
do_anchor_images = True

# -----------------------------------------------------------------------------------
# Directory structure
# -----------------------------------------------------------------------------------

input_tab = Table.read('jwst_image_key.txt', format='ascii.csv',comment='#')

# add a reference table giving the x-axis range of the comparison data

gal_names = np.unique(np.array(input_tab['galaxy']))
filters = np.unique(np.array(input_tab['filter']))

my_input_root = '../working_data/processed_jwst/initial_matched_res/'
my_plot_dir = '../working_data/processed_jwst/background_plots/'
my_table_dir = '../working_data/processed_jwst/background_tables/'
my_external_comp_dir = '../orig_data/background_comps/'

just_targets = []

ref_filt_miri = 'F770W'
ref_filt_nircam = 'F300M'

ext_comp_filt_miri = 'F770W'
ext_comp_filt_nircam = 'F300M'

# -----------------------------------------------------------------------------------
# Loop over all galaxies and do the comparison between bands
# -----------------------------------------------------------------------------------

full_results_dict = []

for this_gal in gal_names:

    if do_compare_all_bands == False:
        continue
    
    if len(just_targets) > 0:
        if this_gal not in just_targets:
            continue

    this_results_dict = []
        
    print("Background comparisons for: ", this_gal)

    # ------------------------------------------------------------
    # MIRI vs MIRI
    # ------------------------------------------------------------    

    print("... MIRI vs MIRI")
        
    for comp_filt in ['F1000W','F1130W','F2100W']:

        # The reference file
        ref_filt = ref_filt_miri
        ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atF2100W.fits'
        
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue
        
        # The comparison file
        comp_file_name = my_input_root + this_gal + '_'+comp_filt+'_atF2100W.fits'

        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu = fits.open(comp_file_name)['SCI']            
            
        # Compare the two images
        this_plot_fname = my_plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        # Compile the results and save them to a dictionary        
        this_result = {'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                       'slope':slope,'intercept':intercept}
        this_results_dict.append(this_result)
        full_results_dict.append(this_result)
        
    # ------------------------------------------------------------
    # MIRI vs External
    # ------------------------------------------------------------

    print("... MIRI vs external")
    
    for ref_filt in ['irac4']:

        # The reference file        
        ref_file_name = my_external_comp_dir + this_gal + '_'+ref_filt+'_atGauss4.fits'
        
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        # Name of file to compare
        comp_filt = ext_comp_filt_miri
        comp_file_name = my_input_root + this_gal + '_'+comp_filt+'_atGauss4.fits'
        
        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu_in = fits.open(comp_file_name)[0]            

        # Align the comparison to the reference image
        comp_hdu = align_image(comp_hdu_in, ref_hdu.header, hdu_in=0,
                               order='bilinear', missing_value=np.nan,
                               outfile=None, overwrite=True)
        
        # Compare the aligned images
        this_plot_fname = my_plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        # Compile the results and save them to a dictionary
        this_result = {'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                       'slope':slope,'intercept':intercept}
        this_results_dict.append(this_result)
        full_results_dict.append(this_result)

            
    for ref_filt in ['w3']:

        # Comparison file
        ref_file_name = my_external_comp_dir + this_gal + '_'+ref_filt+'_atGauss15.fits'

        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        # Name of file to compare
        compt_filt = ext_comp_filt_miri
        comp_file_name = my_input_root + this_gal + '_'+comp_filt+'_atGauss4.fits'
        
        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu_in = fits.open(comp_file_name)[0]            

        comp_hdu = align_image(comp_hdu_in, ref_hdu.header, hdu_in=0,
                               order='bilinear', missing_value=np.nan,
                               outfile=None, overwrite=True)
        
        # Compare
        this_plot_fname = my_plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        # Compile the results and save them to a dictionary
        this_result = {'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                       'slope':slope,'intercept':intercept}
        this_results_dict.append(this_result)
        full_results_dict.append(this_result)
        
    # ------------------------------------------------------------        
    # NIRCAM vs NIRCAM
    # ------------------------------------------------------------    

    print("... NIRCam vs NIRCam")
    
    for comp_filt in ['F200W','F335M','F360M']:

        # Identify the reference image
        ref_filt = ref_filt_nircam
        ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atF2100W.fits'

        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        # Identify the comparison image
        comp_file_name = my_input_root + this_gal + '_'+comp_filt+'_atF2100W.fits'
        
        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu = fits.open(comp_file_name)['SCI']            
            
        # Compare
        this_plot_fname = my_plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        # Compile the results and save them to a dictionary
        this_result = {'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                   'slope':slope,'intercept':intercept}
        this_results_dict.append(this_result)
        full_results_dict.append(this_result)
        
    # ------------------------------------------------------------        
    # NIRCAM vs External
    # ------------------------------------------------------------    

    print("... NIRCam vs external")
    
    ref_filt = 'F300M'
    ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atGauss7p5.fits'
        
    for comp_filt in ['irac1','irac2','w1', 'w2']:

        # Select the relevant table row
        comp_file_name = my_external_comp_dir + this_gal + '_'+comp_filt+'_atGauss7p5.fits'

        # Check that the files are present
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu_in = fits.open(comp_file_name)[0]            

        # Align to the reference
        comp_hdu = align_image(comp_hdu_in, ref_hdu.header, hdu_in=0,
                               order='bilinear', missing_value=np.nan,
                               outfile=None, overwrite=True)
        
        # Compare
        this_plot_fname = my_plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        # Compile the results and save them to a dictionary
        this_result = {'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                   'slope':slope,'intercept':intercept}
        this_results_dict.append(this_result)
        full_results_dict.append(this_result)
        
    # ------------------------------------------------------------        
    # Output to disk for one galaxy
    # ------------------------------------------------------------    
        
    print("... writing this galaxy to disk")
        
    this_table = Table(this_results_dict)
    this_table.write(my_table_dir+this_gal+'_background_comps.ecsv', overwrite=True)

    print("")
    
# -----------------------------------------------------------------------------------
# Output to disk for all galaxies
# -----------------------------------------------------------------------------------

print("... writing all galaxies to disk")
full_table = Table(full_results_dict)
full_table.write(my_table_dir+'all_galaxies_background_comps.ecsv', overwrite=True)
