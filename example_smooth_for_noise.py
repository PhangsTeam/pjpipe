import os
import socket
import glob

from utils_jwst import *

from astropy.table import Table
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.stats import mad_std

# This module holds most of the key routines

from utils_jwst import *

# Open the 21 micron file
hdu21micron = fits.open('../working_data/stitched/ngc0628/ngc0628_miri_f2100w_anchored_at2100.fits')[0]

# Make a 2 arcsecond FWHM PSF
kernel_hdu = make_gaussian_psf(fwhm_arcsec = 2.0, oversample_by=20., outfile=None)

# Smooth and subtract
smoothed_hdu = conv_with_kernel(hdu21micron, kernel_hdu, outfile=None)
diff_map = hdu21micron.data - smoothed_hdu.data

# Look at the result
plt.clf()
plt.imshow(diff_map, vmin=-1, vmax=1)
plt.show()

# Make a histogram
plt.clf()
plt.hist(np.ravel(diff_map), range=(-5,5.), bins=100)
plt.show()

# Note the noise
rms = mad_std(diff_map, ignore_nan = True)
print("Approximate noise: ", rms)
