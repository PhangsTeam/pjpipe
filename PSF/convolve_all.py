#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 19:33:00 2022

@author: belfiore
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
import copy

import webbpsf
from make_kernels import resample, get_pixscale
from astropy.convolution import convolve

# directory where the kernels are saved
kernel_dir = '/Volumes/fbdata2/CODE/JWST/jwst_scripts/PSF/kernels/'

# directory where the reduced and aligned JWST files are 
data_dir = '/Volumes/fbdata2/DATA/PHANGS/JWST/convolved/'

# list of the PHANGS-JWST filters, others can be added if necessary
nircam_psfs = [
    'F200W',
    'F300M',
    'F335M',
    'F360M',
]

miri_psfs = [
    'F770W',
    'F1000W',
    'F1130W',
    'F2100W',
]

VERSION = 'DR1'
galaxies =[ 'IC5332', 'NGC1365', 'NGC7496']
target_name = 'Gauss_{:.3f}'.format(0.89)
# %%

# these get_miri and get_nircam utility functions are macine dependant
#, I did not have time to figure out 
# where the files are saved on astronodes or the folder structure generated
# by the pipeline
def get_miri(gal_name, band):
    miri = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/MIRI_'+VERSION+'/'+gal_name+'/'+
                         gal_name.lower()+'_miri_'+band+'_anchored.fits')
    image=miri[0].data
    header = miri[0].header
    
    error = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/MIRI_'+VERSION+'/'+gal_name+'/'+
                      gal_name.lower()+'_miri_'+band+'_noisemap.fits')[0].data
    return(image, error, header)


def get_nircam(gal_name, band):
    nircam = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/NIRCam_'+VERSION+'/'+gal_name+'/'+
                         gal_name.lower()+'_nircam_lv3_'+band.lower()+'_i2d_align.fits')
    image=nircam["SCI"].data
    error = nircam['ERR'].data
    header = nircam[1].header
    return(image, error, header)
# %%
miri_psfs_n = copy.copy(miri_psfs)
miri_psfs_n.remove('F2100W')
all_PSFs = nircam_psfs+ miri_psfs_n
all_cameras = ['NIRCam']*len(nircam_psfs) + ['MIRI']*len(miri_psfs_n)

galaxies_nircam =['NGC0628', 'NGC1365', 'NGC7496']
galaxies_miri =['NGC0628', 'NGC1365', 'NGC7496', 'IC5332']

for ii in range(len(all_PSFs)):
        print( all_cameras[ii], all_PSFs[ii], ' to F2100W')
        input_filter = {'camera':all_cameras[ii], 'filter':all_PSFs[ii]}
        target_filter = {'camera':'MIRI', 'filter':'F2100W'}
        
        if input_filter['camera']=='NIRCam': 
            galaxies_iterate =galaxies_nircam
        elif input_filter['camera']=='MIRI': 
            galaxies_iterate =galaxies_miri
            
        #loop over galaxies
        for gal_name in galaxies_iterate:
         
            band = input_filter['filter']
            print(gal_name, band)
            
            # get the images
            if input_filter['camera']=='NIRCam': 
                image, error, header = get_nircam(gal_name, band)
            elif input_filter['camera']=='MIRI': 
                image, error, header = get_miri(gal_name, band)
            
            target_pixscale = get_pixscale(header)
        
            
            print('kernel reading')
            kernel_name = kernel_dir+'%s_to_%s.fits' % (input_filter['filter'], 
                                                        target_filter['filter'])
            kernel = fits.open(kernel_name)
            kernel = resample(kernel[0].data, get_pixscale(kernel[0]), target_pixscale)
            print(kernel.shape,target_pixscale)
            print('convolving data')
            target_conv = convolve(image, kernel, preserve_nan=True, fill_value=np.nan)
            print('convolving  error')
            target_err_conv = np.sqrt(convolve(error**2, kernel**2, 
                        preserve_nan=True, boundary=None, normalize_kernel=False))
            #target_err_conv = target_conv+np.nan
            
            print('saving')
            header['PSF']=target_filter['filter']
            hdul = fits.HDUList([
                 fits.PrimaryHDU(),
                 fits.ImageHDU(data=np.array(target_conv, dtype=np.float32),
                               name='SCI', header=header),
                 fits.ImageHDU(data=np.array(header, dtype=np.float32),
                               name='ERR', header=header)])
                 #fits.ImageHDU(data=np.array(bkg_image, dtype=np.float32),name='BKG', header=miriheader)])
            
            
            file_name = data_dir+gal_name.lower()+'_'+band.upper()+'_convolvedto_'+target_filter['filter']+'.fits'
            hdul.writeto(file_name, overwrite=True)
        
# %%
# this is just a test
# for gal_name in [galaxies[1]]:
#     for band in [nircam_psfs[0]]:
#         print(gal_name, band)
        
#         # all of this is machine dependent, I did not have time to figure out 
#         # where the files are saved on astronodes or the folder structure generated
#         # by the pipeline
        
        
#         print(band, target_name)
        
#         print('kernel reading')
#         kernel_name = kernel_dir+'%s_to_%s.fits' % (band, target_name)
#         kernel = fits.open(kernel_name)
#         kernel = resample(kernel[0].data, get_pixscale(kernel[0]), target_pixscale)
#         print(kernel.shape,target_pixscale)
#         print('convolving data')
#         target_conv = convolve(image, kernel, preserve_nan=True, fill_value=np.nan)
#         print('convolving  error')
#         target_err_conv = np.sqrt(convolve(error**2, kernel**2, 
#                     preserve_nan=True, boundary=None, normalize_kernel=False))
#         #target_err_conv = target_conv+np.nan
        
#         print('saving')
#         hdul = fits.HDUList([
#              fits.PrimaryHDU(),
#              fits.ImageHDU(data=np.array(target_conv, dtype=np.float32),name='SCI', header=miriheader),
#              fits.ImageHDU(data=np.array(target_err_conv, dtype=np.float32),name='ERR', header=miriheader)])
#              #fits.ImageHDU(data=np.array(bkg_image, dtype=np.float32),name='BKG', header=miriheader)])
        
        
#         file_name = data_dir+gal_name.lower()+'_'+band.upper()+'_'+VERSION+target_name+'TEST.fits'
#         hdul.writeto(file_name, overwrite=True)