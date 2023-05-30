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

# This module holds the control flow tags

from postprocess_control_flow import *

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Loop over all galaxies and do the comparison between bands
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

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
        ref_file_name = my_initial_matched_res_dir + \
            this_gal + '_'+ref_filt+'_atF2100W.fits'
        
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue
        
        # The comparison file
        comp_file_name = my_initial_matched_res_dir + \
            this_gal + '_'+comp_filt+'_atF2100W.fits'

        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu = fits.open(comp_file_name)['SCI']            
            
        # Compare the two images
        this_plot_fname = my_plot_dir+this_gal+'_' + \
            comp_filt+'_vs_'+ref_filt+'.png'
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
        ref_file_name = my_external_comp_dir + \
            this_gal + '_'+ref_filt+'_atGauss4.fits'
        
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        # Name of file to compare
        comp_filt = ext_comp_filt_miri
        comp_file_name = my_initial_matched_res_dir + \
            this_gal + '_'+comp_filt+'_atGauss4.fits'
        
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
        ref_file_name = my_external_comp_dir + \
            this_gal + '_'+ref_filt+'_atGauss15.fits'

        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        # Name of file to compare
        comp_filt = ext_comp_filt_miri
        comp_file_name = my_initial_matched_res_dir + \
            this_gal + '_'+comp_filt+'_atGauss4.fits'
        
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
            # Note WISE3 specific
            xmin=0.0, xmax=0.8, binsize=0.1,
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
        ref_file_name = my_initial_matched_res_dir + \
            this_gal + '_'+ref_filt+'_atF2100W.fits'

        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        # Identify the comparison image
        comp_file_name = my_initial_matched_res_dir + \
            this_gal + '_'+comp_filt+'_atF2100W.fits'
        
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
            
    for ref_filt in ['irac1','irac2','w1', 'w2']:

        # NIRCam band to compare
        comp_filt = ext_comp_filt_nircam
        comp_file_name = my_initial_matched_res_dir + \
            this_gal + '_'+comp_filt+'_atGauss7p5.fits'
        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue

        # External reference file
        ref_file_name = my_external_comp_dir + this_gal + \
            '_'+ref_filt+'_atGauss7p5.fits'        
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
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

if do_compare_all_bands:
    print("... writing all galaxies to disk")
    full_table = Table(full_results_dict)
    full_table.write(my_table_dir+'all_galaxies_background_comps.ecsv', overwrite=True)

# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%
# Loop over all galaxies, solve for offsets, and apply offsets
# &%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%&%

for this_gal in gal_names:

    if do_anchor_images == False:
        continue

    this_table_name = my_table_dir+this_gal+'_background_comps.ecsv'
    if os.path.isfile(this_table_name) == False:
        print("Needed table file not found: ", this_table_name)
    this_table = Table.read(this_table_name, format='ascii')
    
    # -----------------------------------------------------------------------------------
    # Work out offsets to apply from tables
    # -----------------------------------------------------------------------------------

    offsets = {}
    
    # MIRI - F770W first
    table_mask = (this_table['galaxy'] == this_gal)* \
        (this_table['comp_filt'] == 'F770W')* \
        (this_table['ref_filt'] == 'irac4')
    if np.sum(table_mask) == 0:
        table_mask = (this_table['galaxy'] == this_gal)* \
            (this_table['comp_filt'] == 'F770W')* \
            (this_table['ref_filt'] == 'w3')
    if np.sum(table_mask) != 0:        
        offsets['F770W'] = -1.0*float((this_table[table_mask]['intercept'])[0])

        for other_miri in ['F1000W','F1130W','F2100W']:

            table_mask = (this_table['galaxy'] == this_gal)* \
                (this_table['comp_filt'] == other_miri)* \
                (this_table['ref_filt'] == 'F770W')
            if np.sum(table_mask) == 0:
                print("Not found filter: ", other_miri)
                continue
            offsets[other_miri] = -1.*float((this_table[table_mask]['intercept'])[0]) \
                + float((this_table[table_mask]['slope'])[0])*offsets['F770W']
    else:
        print("No F770W for ", this_gal)
        
    # NIRCam - F300M first    
    table_mask = (this_table['galaxy'] == this_gal)* \
        (this_table['comp_filt'] == 'F300M')* \
        (this_table['ref_filt'] == 'irac1')
    if np.sum(table_mask) == 0:
        table_mask = (this_table['galaxy'] == this_gal)* \
            (this_table['comp_filt'] == 'F300M')* \
            (this_table['ref_filt'] == 'w1')
    if np.sum(table_mask) != 0:        
        offsets['F300M'] = -1.0*float((this_table[table_mask]['intercept'])[0])

        for other_nircam in ['F200W','F335M','F360M']:

            table_mask = (this_table['galaxy'] == this_gal)* \
                (this_table['comp_filt'] == other_nircam)* \
                (this_table['ref_filt'] == 'F300M')
            if np.sum(table_mask) == 0:
                print("Not found filter: ", other_nircam)
                continue
            offsets[other_nircam] = -1.*float((this_table[table_mask]['intercept'])[0]) \
                + float((this_table[table_mask]['slope'])[0])*offsets['F300M']
    else:
        print("No F300M for ", this_gal)
            
    print("For galaxy: ", this_gal)            
    for this_filt in offsets.keys():
        print(this_filt, offsets[this_filt])
    print("")
        
    # -----------------------------------------------------------------------------------
    # Apply the offsets and rewrite to disk
    # -----------------------------------------------------------------------------------

    for this_filt in filters:

        # Select the relevant table row
        if np.sum((image_key['galaxy'] == this_gal)*(image_key['filter'] == this_filt)) == 0:
            print("No match for: ", this_gal, this_filt)
            continue
        tab_mask = (image_key['galaxy'] == this_gal)*(image_key['filter'] == this_filt)
        input_file = my_input_root+ \
            str(np.array(image_key[tab_mask]['filename'])[0])

        # Check that the input file is present
        if os.path.isfile(input_file) == False:
            print("Input file not found, skipping. ", input_file)
            continue

        # Read the input science image
        input_hdu = fits.open(input_file)['SCI']

        this_offset = float(offsets[this_filt])
        input_hdu.header['BKGRDVAL'] = this_offset
        new_data = input_hdu.data
        new_data[new_data==0] = np.nan
        new_data = new_data + this_offset
        input_hdu.data = new_data
        
        # Write to disk
        output_file_name = my_anchored_dir + this_gal + '_'+this_filt+'_anchored.fits'    
        input_hdu.writeto(output_file_name, overwrite=True)
