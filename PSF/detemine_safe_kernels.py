#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 18:10:43 2023

@author: belfiore
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits, ascii
import astropy.table as table
import make_kernels as ker

def makeGaussian_1D(x, mu, sig, normalise=False):
    gauss = np.exp(-np.power((x - mu)/sig, 2.)/2)
    if normalise==True: gauss =gauss *1./(np.sqrt(2.*np.pi)*sig)
    return gauss


def makeGaussian_2D(X, M, S, normalise=False):
    gauss = np.exp(-np.power((X[0] - M[0])/S[0], 2.)/2)*np.exp(-np.power((X[1] - M[1])/S[1], 2.)/2)
    if normalise==True: gauss =gauss *1./(2.*np.pi*S[0]*S[1])
    return gauss


def evaluate_kernel(kk):
    from astropy.convolution import convolve
    kk.kernel=kk.kernel/np.sum(kk.kernel)
    target_conv = convolve(kk.source_psf, kk.kernel)
    # D kernel performance measure Aniano Eq 20
    D = np.sum(np.abs(target_conv-kk.target_psf))
    # Wm kernel performance measure Aniano eq 21
    Wm = 0.5 *np.sum( np.abs(kk.kernel) - kk.kernel)
    return D, Wm

# Do a systematic search of the best Gaussian kernel by exploring kernels up to FWHM_Gauss = [1.1-1.5]*FWHM_source_PSF
# Here we calcualted 11 kernels

#input directory where you have saved the JWST PSFs
psf_dir = '/Volumes/fbdata3/CODE/JWST/jwst_scripts/PSF/PSF/'

input_filter = {'camera':'MIRI', 'filter':'F2100W'}

# read the source JWST kernel
# psf = fits.open(PSF_DIR+aa['instr']+'/PSF_'+aa['instr']+'_predicted_opd_filter_'+aa['ch']+'.fits')
# source_pixscale = psf[0].header['PIXELSCL']
# source_psf=psf[0].data



source_psf_path = psf_dir+input_filter['camera']+'_PSF_filter_'+\
            input_filter['filter']+'.fits'
source_psf = fits.open(source_psf_path)[0]
source_pixscale = source_psf.header['PIXELSCL']
source_fwhm = ker.fit_2d_gaussian(source_psf.data, pixscale=source_pixscale)

common_pixscale=source_pixscale
target_pixscale= source_pixscale
    

factor = np.linspace(1.04, 1.5, 11)
D_v, Wm_v = np.zeros(len(factor)), np.zeros(len(factor))

for ii, ff in enumerate(factor):
    print(ii, ff)
    yy, xx = np.meshgrid(np.arange(361)-180,np.arange(361)-180 )
    target_gaussian = {'fwhm':source_fwhm*ff}
    target_psf = makeGaussian_2D((xx, yy), (0,0), (
     target_gaussian['fwhm']/2.355/target_pixscale, \
         target_gaussian['fwhm']/2.355/target_pixscale) )


    grid_size_arcsec = np.array([331 * target_pixscale,
                                 331 * target_pixscale])

    kk = ker.MakeConvolutionKernel(source_psf=source_psf,
                               source_pixscale=source_pixscale,
                               source_name=input_filter['filter'],
                               target_psf=target_psf,
                               target_pixscale=target_pixscale,
                               target_name= 'Gauss_{:.3f}'.format(target_gaussian['fwhm']),
                               common_pixscale = common_pixscale,
                               grid_size_arcsec =grid_size_arcsec
                               )
    kk.make_convolution_kernel()
    D_v[ii], Wm_v[ii] = evaluate_kernel(kk)
    print(D_v, Wm_v)
    
#%%
#Evaluate the kernels by calculating their D and Wm
# target_gaussian_fwhm = factor*source_fwhm

# for ii in range(len(factor)):
#     kernel_name = kernel_dir+'%s_to_%s.fits' % (input_filter['filter'], 
#                                                             'Gauss_{:.3f}'.format(target_gaussian_fwhm[ii]))
#     kernel = fits.open(kernel_name)
#     D_v[ii], Wm_v[ii] = evaluate_kernel(kk)


# %%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,4))
target_fwhm_v = factor*source_fwhm
ax1.plot(target_fwhm_v, D_v, label='D', lw=4)
ax1.set_xlabel("Gaussian FWHM")
ax1.set_ylabel("D")
ax1.legend()
ax2.plot(target_fwhm_v, Wm_v, label='W', lw=4)
ax2.set_xlabel("Gaussian FWHM")
ax2.set_ylabel(r"$W_{-}$")

ax2.axhline(y=1, ls='--', c='k')
ax2.axhline(y=0.5, ls='--', c='k')
ax2.axhline(y=0.3, ls='--', c='k')
ax2.legend()

out = np.interp(np.array([0.3, 0.5, 1.0]), Wm_v[::-1], target_fwhm_v[::-1])

print('Wm, very safe {:.2f}", safe {:.2f}", aggressive {:.2f}", source {:.2f}" '.format( 
    *out, source_fwhm))

out2 = np.interp(out,factor*source_fwhm, D_v )
print('D, very safe {:.2f}, safe {:.2f}, aggressive {:.2f}'.format( *out2))
for ii in range(len(out2)):
    ax1.axvline(x=out[ii], ls='--', c='k')
    ax1.text(out[ii], 0.11, '{:.2f}"'.format(out[ii]))