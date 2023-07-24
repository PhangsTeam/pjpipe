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
from reproject.mosaicking import find_optimal_celestial_wcs

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
# target_name = 'Gauss_{:.3f}'.format(0.89)

def get_copt_fwhm(gal_name):
    """For a given PHANGS galaxy, get the FWHM of the copt MUSE data
    """

    t= table.Table.read('muse_dr2_v1.fits')
    ii = t['name']==gal_name
    copt_fwhm = float(t[ii]['muse_copt_FWHM'])
    return copt_fwhm

def _make_fake(image, size=1000):
    cenx, ceny = int(image.shape[1]/2), int(image.shape[0]/2)
    small =image[ceny-size:ceny+size, cenx-size:cenx+size]
    return(small)

# nircam_pixel_scale = 0.0630
# miri_pixel_scale = 0.110
# %%

# these get_miri and get_nircam utility functions are machine dependant
#, I did not have time to figure out 
# where the files are saved on astronodes or the folder structure generated
# by the pipeline

def get_miri_hdu(gal_name, band):
    hdu = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/MIRI_'+VERSION+'/'+gal_name+'/'+
                         gal_name.lower()+'_miri_'+band+'_anchored.fits')
    return(hdu)

def get_nircam_hdu(gal_name, band):
    hdu = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/NIRCam_'+VERSION+'/'+gal_name+'/'+
                         gal_name.lower()+'_nircam_lv3_'+band.lower()+'_i2d_align.fits')
    return(hdu)
    
def get_miri(gal_name, band):
    miri = get_miri_hdu(gal_name, band)
    image=miri[0].data
    header = miri[0].header
    # the errors are saved in a separate file for the DR1 (letter-version) reductions
    error = fits.open('/Volumes/fbdata2/DATA/PHANGS/JWST/MIRI_'+VERSION+'/'+gal_name+'/'+
                      gal_name.lower()+'_miri_'+band+'_noisemap.fits')[0].data
    return(image, error, header)


def get_nircam(gal_name, band):
    nircam = get_nircam_hdu(gal_name, band)
    image=nircam["SCI"].data
    error = nircam['ERR'].data
    header = nircam[1].header
    return(image, error, header)
# %%

# SOME NONSENSE TO BE DEBUGGED LATER

# galaxies_nircam =['NGC0628', 'NGC1365', 'NGC7496']
# galaxies_miri =['NGC0628', 'NGC1365', 'NGC7496', 'IC5332']
# #everything to 2100
# all_PSFs = nircam_psfs+ miri_psfs
# all_cameras = ['NIRCam']*len(nircam_psfs) + ['MIRI']*len(miri_psfs)
# hdu_list_dict={'galaxy':galaxies_miri, 'hdu_list':[[]]*4}
# # define a final HDU
# for jj, galaxy in enumerate(galaxies_miri):
#     hdu_list=[]
#     camera = []
#     print(jj, galaxy)
#     for ii, band in enumerate(all_PSFs):
#         print()
#         if all_cameras[ii]=='NIRCam':
#             try:
#                 hdu = get_nircam_hdu(galaxy, band)[1]
#                 camera= camera+['NIRCam']
#             except:
#                 print('non NIRCam data for '+galaxy+ ' '+ band)

            
#         if all_cameras[ii]=='MIRI':
#             hdu = get_miri_hdu(galaxy, band)[0]
#             camera= camera+['MIRI']
#         print(hdu_list)
#         hdu_list =hdu_list+[hdu]
#     print(camera)
        
#     hdu_list_dict['hdu_list'][jj] =hdu_list
    
# wcs_out, shape_out = find_optimal_celestial_wcs(list_of_hdus)
#%%         # 
galaxies_nircam =['NGC0628', 'NGC1365', 'NGC7496']
galaxies_miri =['NGC0628', 'NGC1365', 'NGC7496', 'IC5332']   
miri_psfs_n = copy.copy(miri_psfs)
miri_psfs_n.remove('F2100W')
all_PSFs = nircam_psfs+ miri_psfs_n
all_cameras = ['NIRCam']*len(nircam_psfs) + ['MIRI']*len(miri_psfs_n)      


# %%

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
        
            input_pixscale = get_pixscale(header)
    
        
            print('kernel reading')
            kernel_name = kernel_dir+'%s_to_%s.fits' % (input_filter['filter'], 
                                                        target_filter['filter'])
            kernel = fits.open(kernel_name)
            kernel = resample(kernel[0].data, get_pixscale(kernel[0]), input_pixscale)
            # need to normalise the kernel to get the correct maths when convolving the errors with ker**2
            kernel = kernel/np.sum(kernel)
            print(kernel.shape,input_pixscale)
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
                 fits.ImageHDU(data=np.array(target_err_conv, dtype=np.float32),
                               name='ERR', header=header)])
                 #fits.ImageHDU(data=np.array(bkg_image, dtype=np.float32),name='BKG', header=miriheader)])
            
            
            file_name = data_dir+gal_name.lower()+'_'+band.upper()+'_convolvedto_'+target_filter['filter']+'.fits'
            hdul.writeto(file_name, overwrite=True)
# %%
#everything to copt

all_PSFs = nircam_psfs+ miri_psfs
all_cameras = ['NIRCam']*len(nircam_psfs) + ['MIRI']*len(miri_psfs)
galaxies_nircam =['NGC0628', 'NGC1365', 'NGC7496']
galaxies_miri =['NGC0628', 'NGC1365', 'NGC7496', 'IC5332']

# debugging
# all_PSFs = [all_PSFs[0]]
# all_cameras = [all_cameras[0]]
# galaxies_nircam = galaxies_nircam[1:3]


for ii in range(len(all_PSFs)):
        print( all_cameras[ii], all_PSFs[ii], ' to copt')
        
        input_filter = {'camera':all_cameras[ii], 'filter':all_PSFs[ii]}
        
        if input_filter['camera']=='NIRCam': 
            galaxies_iterate =galaxies_nircam
        elif input_filter['camera']=='MIRI': 
            galaxies_iterate =galaxies_miri
            
        #loop over galaxies
        for gal_name in galaxies_iterate:
            
            copt_psf =  get_copt_fwhm(gal_name)
            target_gaussian = {'fwhm':copt_psf}
         
            band = input_filter['filter']
            print(gal_name, band, copt_psf)

            # get the images
            if input_filter['camera']=='NIRCam': 
                image, error, header = get_nircam(gal_name, band)
            elif input_filter['camera']=='MIRI': 
                image, error, header = get_miri(gal_name, band)
            
            input_pixscale = get_pixscale(header)
            print(input_pixscale)

            print('kernel reading')
            kernel_name = kernel_dir+'%s_to_%s.fits' % (input_filter['filter'], 
                                                        'Gauss_{:.3f}'.format(target_gaussian['fwhm']))
            kernel = fits.open(kernel_name)
            print(get_pixscale(kernel[0]))
            kernel = resample(kernel[0].data, get_pixscale(kernel[0]), input_pixscale)
            # need to normalise the kernel to get the correct maths when convolving the errors with ker**2
            kernel = kernel/np.sum(kernel)
            print(kernel.shape,input_pixscale)

            print('convolving data')
            target_conv = convolve(image, kernel, preserve_nan=True, fill_value=np.nan)
            print('convolving  error')
            target_err_conv = np.sqrt(convolve(error**2, kernel**2, 
                        preserve_nan=True, boundary=None, normalize_kernel=False))
            #target_err_conv = target_conv+np.nan
            
            print('saving')
            header['PSF']='Gauss_{:.3f}'.format(target_gaussian['fwhm'])
            hdul = fits.HDUList([
                 fits.PrimaryHDU(),
                 fits.ImageHDU(data=np.array(target_conv, dtype=np.float32),
                               name='SCI', header=header),
                 fits.ImageHDU(data=np.array(target_err_conv, dtype=np.float32),
                               name='ERR', header=header)])
                 #fits.ImageHDU(data=np.array(bkg_image, dtype=np.float32),name='BKG', header=miriheader)])
            
            
            file_name = data_dir+gal_name.lower()+'_'+band.upper()+'_convolvedto_'+'Gauss_{:.3f}'.format(target_gaussian['fwhm'])+'.fits'
            hdul.writeto(file_name, overwrite=True)
# %%




# from astropy.wcs import WCS as WCS_astropy
# wcs_out = WCS_astropy(header)
# new_header = wcs_out.to_header()
# new_header['PSF']='Gauss_{:.3f}'.format(target_gaussian['fwhm'])

# hdul = fits.HDUList([fits.PrimaryHDU(), 
#     fits.ImageHDU(data=np.array(target_conv, dtype=np.float32),name='SCI', header=new_header),
#     fits.ImageHDU(data=np.array(target_err_conv, dtype=np.float32), name='ERR', header=new_header)])


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
        # hdul.writeto(file_name, overwrite=True)
        
# %%

# all_PSFs = nircam_psfs+ miri_psfs
# all_cameras = ['NIRCam']*len(nircam_psfs) + ['MIRI']*len(miri_psfs)
# galaxies_nircam =['NGC0628', 'NGC1365', 'NGC7496']
# galaxies_miri =['NGC0628', 'NGC1365', 'NGC7496', 'IC5332']

# # debugging
# # all_PSFs = [all_PSFs[0]]
# # all_cameras = [all_cameras[0]]
# # galaxies_nircam = galaxies_nircam[1:3]


# for ii in range(len(all_PSFs)):
#         print( all_cameras[ii], all_PSFs[ii], ' to copt')
        
#         input_filter = {'camera':all_cameras[ii], 'filter':all_PSFs[ii]}
        
#         if input_filter['camera']=='NIRCam': 
#             galaxies_iterate =galaxies_nircam
#         elif input_filter['camera']=='MIRI': 
#             galaxies_iterate =galaxies_miri
            
#         #loop over galaxies
#         for gal_name in galaxies_iterate:
            
#             copt_psf =  get_copt_fwhm(gal_name)
#             target_gaussian = {'fwhm':copt_psf}
         
#             band = input_filter['filter']
     

#             # get the images
#             if input_filter['camera']=='NIRCam': 
#                 image, error, header = get_nircam(gal_name, band)
#             elif input_filter['camera']=='MIRI': 
#                 image, error, header = get_miri(gal_name, band)
            
#             input_pixscale = get_pixscale(header)
 

  
#             kernel_name = kernel_dir+'%s_to_%s.fits' % (input_filter['filter'], 
#                                                         'Gauss_{:.3f}'.format(target_gaussian['fwhm']))
#             kernel = fits.open(kernel_name)

#             kernel = resample(kernel[0].data, get_pixscale(kernel[0]), input_pixscale)
#             print(np.sum(kernel))
#             # kernel=kernel/np.sum(kernel)
            
#             file_name = data_dir+gal_name.lower()+'_'+band.upper()+\
#                 '_convolvedto_'+'Gauss_{:.3f}'.format(target_gaussian['fwhm'])+'.fits'
                
#             hdu = fits.open(file_name)
#             hdu['ERR'].data = hdu['ERR'].data/np.sum(kernel)
            
#             # file_name2 = data_dir+gal_name.lower()+'_'+band.upper()+\
#             #     '_convolvedto_'+'Gauss_{:.3f}'.format(target_gaussian['fwhm'])+'_errcorr.fits'
#             hdu.writeto(file_name, overwrite=True)