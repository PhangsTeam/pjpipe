import os
import socket
import glob

from astropy.table import Table
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.convolution import convolve_fft, Gaussian2DKernel
from scipy.interpolate import RegularGridInterpolator
from reproject import reproject_interp
import matplotlib.pyplot as plt

from utils_jwst import *

input_tab = Table.read('jwst_image_key.txt', format='ascii.csv',comment='#')

gal_names = np.unique(np.array(input_tab['galaxy']))
filters = np.unique(np.array(input_tab['filter']))

my_input_root = '../working_data/matched_res/'
plot_dir = '../plots/background_plots/'
external_comp_dir = '../orig_data/background_comps/'

results_dict = []

for this_gal in gal_names:

    print("Background comparisons for: ", this_gal)

    # ------------------------------------------------------------
    # MIRI vs MIRI
    # ------------------------------------------------------------    
    
    ref_filt = 'F770W'
    ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atF2100W.fits'
        
    for comp_filt in ['F1000W','F1130W','F2100W']:

        # Comparison file
        comp_file_name = my_input_root + this_gal + '_'+comp_filt+'_atF2100W.fits'

        # Check that the files are present
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu = fits.open(comp_file_name)['SCI']            
            
        # Compare
        this_plot_fname = plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        results_dict.append({'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                             'slope':slope,'intercept':intercept})
        
    # ------------------------------------------------------------
    # MIRI vs External
    # ------------------------------------------------------------

    ref_filt = 'F770W'
    ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atGauss4.fits'
    
    for comp_filt in ['irac4']:

        # Comparison file
        comp_file_name = external_comp_dir + this_gal + '_'+comp_filt+'_atGauss4.fits'

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
        this_plot_fname = plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        results_dict.append({'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                             'slope':slope,'intercept':intercept})

    ref_filt = 'F770W'
    ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atGauss15.fits'
    
    for comp_filt in ['w3']:

        # Comparison file
        comp_file_name = external_comp_dir + this_gal + '_'+comp_filt+'_atGauss15.fits'

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

        comp_hdu = align_image(comp_hdu_in, ref_hdu.header, hdu_in=0,
                               order='bilinear', missing_value=np.nan,
                               outfile=None, overwrite=True)
        
        # Compare
        this_plot_fname = plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        results_dict.append({'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                             'slope':slope,'intercept':intercept})
        
    # ------------------------------------------------------------        
    # NIRCAM vs NIRCAM
    # ------------------------------------------------------------    
        
    ref_filt = 'F300M'
    ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atF2100W.fits'
        
    for comp_filt in ['F200W','F335M','F360M']:

        # Select the relevant table row
        comp_file_name = my_input_root + this_gal + '_'+comp_filt+'_atF2100W.fits'

        # Check that the files are present
        if os.path.isfile(ref_file_name) == False:
            print("Reference file not found, skipping. ", ref_file_name)
            continue

        if os.path.isfile(comp_file_name) == False:
            print("Comparison file not found, skipping. ", comp_file_name)
            continue
        
        # Read the images
        ref_hdu = fits.open(ref_file_name)[0]
        comp_hdu = fits.open(comp_file_name)['SCI']            
            
        # Compare
        this_plot_fname = plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        results_dict.append({'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                             'slope':slope,'intercept':intercept})
        
    # ------------------------------------------------------------        
    # NIRCAM vs External
    # ------------------------------------------------------------    

    ref_filt = 'F300M'
    ref_file_name = my_input_root + this_gal + '_'+ref_filt+'_atGauss7p5.fits'
        
    for comp_filt in ['irac1','irac2','w1', 'w2']:

        # Select the relevant table row
        comp_file_name = external_comp_dir + this_gal + '_'+comp_filt+'_atGauss7p5.fits'

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
        this_plot_fname = plot_dir+this_gal+'_'+comp_filt+'_vs_'+ref_filt+'.png'
        slope, intercept = solve_for_offset(
            comp_hdu, ref_hdu, mask_hdu=None,
            xmin=0.25, xmax=2.0, binsize=0.1,
            make_plot=True, plot_fname=this_plot_fname,
            label_str=this_gal+'\n'+comp_filt+' vs. '+ref_filt)

        results_dict.append({'galaxy':this_gal,'ref_filt':ref_filt,'comp_filt':comp_filt,
                             'slope':slope,'intercept':intercept})

comp_table = Table(results_dict)
comp_table.write('band_comps.ecsv', overwrite=True)
