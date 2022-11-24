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
data_dir = '/Volumes/fbdata2/DATA/PHANGS/JWST/MIRI_DR1/'

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
target_name = 'Gauss_{:.3f}'.format(0.92)


# this is just a test
for gal_name in [galaxies[0]]:
    for band in [miri_psfs[0]]:
        print(gal_name, band)
        
        # all of this is machine dependent, I did not have time to figure out 
        # where the files are saved on astronodes or the folder structure generated
        # by the pipeline
        miri = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/MIRI_'+VERSION+'/'+gal_name+'/'+
                         gal_name.lower()+'_miri_'+band+'_anchored.fits')
        image=miri[0].data
        miriheader = miri[0].header
        target_pixscale = get_pixscale(miri[0])
        error = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/MIRI_'+VERSION+'/'+gal_name+'/'+
                          gal_name.lower()+'_miri_'+band+'_noisemap.fits')[0].data
        
        print(band, target_name)
        
        print('kernel reading')
        kernel_name = kernel_dir+'%s_to_%s.fits' % (band, target_name)
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
        hdul = fits.HDUList([
             fits.PrimaryHDU(),
             fits.ImageHDU(data=np.array(target_conv, dtype=np.float32),name='SCI', header=miriheader),
             fits.ImageHDU(data=np.array(target_err_conv, dtype=np.float32),name='ERR', header=miriheader)])
             #fits.ImageHDU(data=np.array(bkg_image, dtype=np.float32),name='BKG', header=miriheader)])
        
        
        file_name = data_dir+gal_name.lower()+'_'+band.upper()+'_'+VERSION+target_name+'TEST.fits'
        hdul.writeto(file_name, overwrite=True)