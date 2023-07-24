#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Created on Thu Nov 24 14:07:52 2022

This script uses webbpsf to generate a fresh version of the JWST kernels
Useful in the future if webbpsf is updated

Updated to include all copt and a wider set of Gaussian options May 23
2023 and moved that material to a separate set of files.

Control the script through the psf_options_file.py in the same
directory and then run it via python generate_JWST_PSF_and_kernels.py

@author: belfiore

"""

# --------------------------------------------------------
# Setup
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import table
import copy

import webbpsf
from make_kernels import MakeConvolutionKernel, profile
from astropy.convolution import convolve

# AKL - moved this up but it's not clear that it's used
nircam_pixel_scale_small = 0.0310
nircam_pixel_scale_large = 0.0630
miri_pixel_scale = 0.110    

# AKL - import the PSFs, copt targets, etc. from a seaprate file that
# doesn't have to go under version control

from psf_options_file import *

# --------------------------------------------------------
# Routines
# --------------------------------------------------------

def makeGaussian_2D(X, M, S, normalise=False):
    gauss = np.exp(-np.power((X[0] - M[0])/S[0], 2.)/2)*np.exp(-np.power((X[1] - M[1])/S[1], 2.)/2)
    if normalise==True: gauss =gauss *1./(2.*np.pi*S[0]*S[1])
    return gauss

def save_miri_PSF(miri_psfs, output_dir='', **kwargs):
    """Generates MIRI PSF using webbpsf and saves then in output dir
    
    
    Parameters
    ----------
    miri_psfs : list
        list of NIRCam filters.
        
    output_dir: string
        path to the output directory to save the PSF files
 
    """

    oversample_factor = kwargs.pop("oversample_factor", 4)
    detector_oversample = kwargs.pop("oversample_factor", 4)
    fov_arcsec = kwargs.pop("fov_arcsec", 19.98)
  
    for filter1 in miri_psfs:
        print('building PSF '+filter1)
    
        miri = webbpsf.MIRI()
        
        miri.filter = filter1
        psf_array = miri.calc_psf(oversample=oversample_factor,
               detector_oversample=detector_oversample,
               fov_arcsec=fov_arcsec, **kwargs)
        
        psf_array.writeto(output_dir+'MIRI_PSF_filter_'+miri.filter+'.fits',
                          overwrite=True)
    
def save_nircam_PSF(nircam_psfs, output_dir='', **kwargs):
    """Generates NIRCam PSF using webbpsf and saves then in output dir
    
    
    Parameters
    ----------
    nircam_psfs : list
        list of NIRCam filters.
        
    output_dir: string
        path to the output directory to save the PSF files
 
    """
    oversample_factor = kwargs.pop("oversample_factor", 4)
    detector_oversample = kwargs.pop("oversample_factor", 4)
    fov_arcsec = kwargs.pop("fov_arcsec", 10)
    
    for filter1 in nircam_psfs:
        print('building PSF '+filter1)
    
        nircam = webbpsf.NIRCam()
        
        nircam.filter = filter1
        psf_array = nircam.calc_psf(oversample=oversample_factor,
               detector_oversample=detector_oversample,
               fov_arcsec=fov_arcsec, **kwargs)
        
        psf_array.writeto(output_dir+'NIRCam_PSF_filter_'+nircam.filter+'.fits', 
                          overwrite=True)


def save_jwst_cross_kernel(input_filter, target_filter, psf_dir='', outdir=''):
    '''Genrates and saves the kernel necessary to convolve the image taken in a 
    JWST input filter into a JWST output filter. It works for both MIRI and NIRCam.
    

    Parameters
    ----------
    input_filter : dict
        Dictionary containing 'camera' and 'filter' keys
    target_filter : dict
        Dictionary containing 'camera' and 'filter' keys.
    psf_dir : str, optional
        Path to the directory where the JWST PSFs are saved. The default is ''.
    outdir : str, optional
        Path to the directory where the kernels will be saved. The default is ''.

    Returns
    -------
    kk : MakeConvolutionKernel
        Object containing the kernel.
        
    Notes
    -------
    If the necessary JWST PSF is not found in psf_dir, the code will use webbpsf
    to generate the PSF. This requires webbpsf to be installed and the necessary
    files to have been added to the path. For more details see 
    https://webbpsf.readthedocs.io/en/latest/installation.html
    '''
  
  

    source_psf_path = psf_dir+input_filter['camera']+'_PSF_filter_'+\
            input_filter['filter']+'.fits'
    try:
        source_psf = fits.open(source_psf_path)[0]

    except FileNotFoundError:
        print('generating PSF with webbpsf!')
        if input_filter['camera']=='MIRI':
            save_miri_PSF([input_filter['filter']], output_dir=psf_dir)
            source_psf = fits.open(source_psf_path)[0]
            
        if input_filter['camera']=='NIRCam':
            save_nircam_PSF([input_filter['filter']], output_dir=psf_dir)
            source_psf = fits.open(source_psf_path)[0]
            
    source_pixscale = source_psf.header['PIXELSCL']
            
    target_psf_path = psf_dir+target_filter['camera']+'_PSF_filter_'+\
            target_filter['filter']+'.fits'
    try:
        target_psf = fits.open(target_psf_path)[0]
       
    except FileNotFoundError:
        print('generating PSF with webbpsf!')
        if input_filter['camera']=='MIRI':
            save_miri_PSF([target_filter['filter']], output_dir=psf_dir)
            target_psf = fits.open(target_psf_path)[0]
            
        if input_filter['camera']=='NIRCam':
            save_nircam_PSF([target_filter['filter']], output_dir=psf_dir)
            target_psf = fits.open(target_psf_path)[0]
            
    target_pixscale = target_psf.header['PIXELSCL']
    common_pixscale = source_pixscale
        
    grid_size_arcsec = np.array([361 * common_pixscale,
                                 361 * common_pixscale])

    kk = MakeConvolutionKernel(source_psf=source_psf,
                               source_pixscale=source_pixscale,
                               source_name=input_filter['filter'],
                               target_psf=target_psf,
                               target_pixscale=target_pixscale,
                               target_name=target_filter['filter'],
                               common_pixscale = common_pixscale,
                               grid_size_arcsec =grid_size_arcsec,
                               verbose=True
                               )
    kk.make_convolution_kernel()
    kk.write_out_kernel(outdir =outdir )
    return kk
     
def save_kernels_to_Gauss(input_filter, target_gaussian, psf_dir='', outdir=''):
    '''
    

    Parameters
    ----------
    input_filter : TYPE
        DESCRIPTION.
    target_gaussian : TYPE
        DESCRIPTION.
    psf_dir : TYPE, optional
        DESCRIPTION. The default is ''.
    outdir : TYPE, optional
        DESCRIPTION. The default is ''.

    Returns
    -------
    None.

    '''

    source_psf_path = psf_dir+input_filter['camera']+'_PSF_filter_'+\
            input_filter['filter']+'.fits'
    source_psf = fits.open(source_psf_path)[0]
    source_pixscale = source_psf.header['PIXELSCL']
    
    common_pixscale=source_pixscale
    target_pixscale= source_pixscale
    
    yy, xx = np.meshgrid(np.arange(361)-180,np.arange(361)-180 )
  
    target_psf = makeGaussian_2D((xx, yy), (0,0), (
         target_gaussian['fwhm']/2.355/target_pixscale, \
             target_gaussian['fwhm']/2.355/target_pixscale) )
    
    
        
    grid_size_arcsec = np.array([331 * target_pixscale,
                                 331 * target_pixscale])

    kk = MakeConvolutionKernel(source_psf=source_psf,
                               source_pixscale=source_pixscale,
                               source_name=input_filter['filter'],
                               target_psf=target_psf,
                               target_pixscale=target_pixscale,
                               target_name= 'Gauss_{:.3f}'.format(target_gaussian['fwhm']),
                               common_pixscale = common_pixscale,
                               grid_size_arcsec =grid_size_arcsec
                               )
    kk.make_convolution_kernel()
    kk.write_out_kernel(outdir =outdir )
    return kk
        
def get_copt_fwhm(gal_name):
    """For a given PHANGS galaxy, get the FWHM of the copt MUSE data
    """

    t= table.Table.read('muse_dr2_v1.fits')
    ii = t['name']==gal_name
    copt_fwhm = float(t[ii]['muse_copt_FWHM'])
    return copt_fwhm
        
def plot_kernel(kk, save_plot=False, save_dir ='' ):
    """Plots source and target PSF and the kernel
    

    Parameters
    ----------
    kk : MakeConvolutionKernel
        The objects returned by a call of MakeConvolutionKernel or the wrapper
        save_jwst_cross_kernel

    Returns
    -------
    None.

    """
   

    target_conv = convolve(kk.source_psf, kk.kernel)
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(12,4))
    
    
    ax1.imshow(np.log10(kk.source_psf/np.max(kk.source_psf)), vmax=0, vmin=-4);
    ax1.set_title(kk.source_name)
    
    ax2.imshow(np.log10(kk.target_psf/np.max(kk.target_psf)), vmax=0, vmin=-4);
    ax2.set_title(kk.target_name)
    
    extent = int(10*kk.target_fwhm/kk.common_pixscale/2)
    ax3.plot(*profile(kk.source_psf/np.max(kk.source_psf), 
                      bins=np.linspace(0, 6*kk.target_fwhm,extent) ,
                      pixscale=kk.common_pixscale), 
             c='b', label=kk.source_name);
    ax3.plot(*profile(kk.target_psf/np.max(kk.target_psf), 
                      bins=np.linspace(0, 6*kk.target_fwhm, extent),
                      pixscale=kk.common_pixscale),
             c='k', label=kk.target_name, lw=5);
    
    ax3.plot(*profile(kk.kernel/np.max(kk.kernel), 
                      bins=np.linspace(0, 6*kk.target_fwhm, extent),
                      pixscale=kk.common_pixscale),
             c='g', label='kernel');
    ax3.plot(*profile(target_conv/np.max(target_conv), 
                      bins=np.linspace(0, 6*kk.target_fwhm, extent),
                      pixscale=kk.common_pixscale),
             c='r', label='model', ls='-')
    # ax3.plot(*ker.profile( (target_conv-kk.target_psf)/kk.target_psf, bins=np.linspace(0, 180, 100)),
    #          c='r', label='residual', ls='--')
    ax3.legend()
    #ax3.set_yscale('log')
    ax3.set_ylim([-0.1, 1.1])
    ax3.set_xlim([0,6*kk.target_fwhm])
    if save_plot ==True:
        plt.savefig(save_dir+kk.source_name+'_'+kk.target_name+'.png',
                    dpi=300)

# %%

# ------------------------------------------------------------------------
# Script that generates kernels
# ------------------------------------------------------------------------

# Depends on options an dlists inside the "psf_options_file"

if do_kern_to_f2100w:

    # 
    # (1) Loop over all PSFs and calculate a kernel to go to F2100W
    #
    
    miri_psfs_no_f2100w = copy.copy(miri_psf_list)
    miri_psfs_no_f2100w.remove('F2100W')
    all_PSFs = nircam_psf_list+ miri_psfs_no_f2100w
    all_cameras = ['NIRCam']*len(nircam_psf_list) + ['MIRI']*len(miri_psfs_no_f2100w)

    for ii in range(len(all_PSFs)): 
        print("")
        print("---------------------------------------------")
        print( all_cameras[ii], all_PSFs[ii], ' to F2100W')
        print("---------------------------------------------")
        input_filter = {'camera':all_cameras[ii], 'filter':all_PSFs[ii]}
        target_filter = {'camera':'MIRI', 'filter':'F2100W'}
    
        this_kern = save_jwst_cross_kernel(
            input_filter, target_filter,
            psf_dir=output_dir_psf, outdir=output_dir_kernels)
        plot_kernel(this_kern,save_plot=True, save_dir=output_dir_kernels)

if do_kern_to_f360m:
    #
    # (2) Loop over all NIRCam PSFs and calculate a kernel to go to F360W
    #

    nircam_psfs_no_f360m = copy.copy(nircam_psf_list)
    nircam_psfs_no_f360m.remove('F360M')
    all_cameras = ['NIRCam']*len(all_PSFs) 
    
    for ii in range(len(nircam_psfs_no_f360m)):
        print("")
        print("---------------------------------------------")
        print( all_cameras[ii], all_PSFs[ii], ' to F360M')
        print("---------------------------------------------")
        input_filter = {'camera':all_cameras[ii], 'filter':all_PSFs[ii]}
        target_filter = {'camera':'NIRCam', 'filter':'F360M'}
    
        this_kern = save_jwst_cross_kernel(
            input_filter, target_filter,
            psf_dir=output_dir_psf, outdir=output_dir_kernels)
        plot_kernel(this_kern,save_plot=True, save_dir=output_dir_kernels)


if do_kern_to_copt:
    #
    # (3) Create kernels targeting the specific MUSE copt values (as Gaussians)
    #

    #get the copt PSF of the JWST galaxies
    copt_fwhm = np.zeros(len(copt_targets))
    for ii, this_gal in enumerate(copt_targets):
        copt_fwhm[ii] = get_copt_fwhm(this_gal)
        print(this_gal,copt_fwhm[ii]  )

    # Try to move NIRCam and MIRI PSFs to the target res
    all_PSFs = nircam_psf_list+ miri_psf_list
    all_cameras = ['NIRCam']*len(nircam_psf_list) + ['MIRI']*len(miri_psf_list)

    for jj in copt_fwhm:
        for ii in range(len(all_PSFs)):
            print("")
            print("---------------------------------------------")
            print( all_cameras[ii], all_PSFs[ii], ' to Gauss ' + '{:.3f}'.format(jj))
            print("---------------------------------------------")
            input_filter = {'camera':all_cameras[ii], 'filter':all_PSFs[ii]}
            target_gaussian = {'fwhm':jj}
        
            this_kern = save_kernels_to_Gauss(
                input_filter, target_gaussian,
                psf_dir=output_dir_psf, outdir=output_dir_kernels)
            plot_kernel(this_kern,save_plot=True, save_dir=output_dir_kernels)

if do_kern_to_gauss:
    #
    # (4) Create kernels targeting the specific Gaussian FWHM values
    #

    # Try to move NIRCam and MIRI PSFs to the target res
    all_PSFs = nircam_psf_list+ miri_psf_list
    all_cameras = ['NIRCam']*len(nircam_psf_list) + ['MIRI']*len(miri_psf_list)
    
    for jj in gauss_res_list:
        for ii in range(len(all_PSFs)):
            print("")
            print("---------------------------------------------")
            print( all_cameras[ii], all_PSFs[ii], ' to Gauss ' + '{:.3f}'.format(jj))
            print("---------------------------------------------")
            input_filter = {'camera':all_cameras[ii], 'filter':all_PSFs[ii]}
            target_gaussian = {'fwhm':jj}
        
            this_kern = save_kernels_to_Gauss(
                input_filter, target_gaussian,
                psf_dir=output_dir_psf, outdir=output_dir_kernels)
            plot_kernel(this_kern,save_plot=True, save_dir=output_dir_kernels)
    
    
            
# %%
