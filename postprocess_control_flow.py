# This consoldiates control flow and directory names, etc., that the
# user will want to change for the postprocessing.

from astropy.table import Table
import numpy as np

# -----------------------------------------------------------------------------------
# Control Flow
# -----------------------------------------------------------------------------------

# True or False sets what to do

# ... in the initial convolve
do_conv_to_f2100w_initial = True
do_conv_to_gauss_initial = True

# ... in the comparison
do_compare_all_bands = True
do_anchor_images = True

# ... in the post-anchoring convolve
do_conv_to_f2100w_anchored = True
do_conv_to_gauss_anchored = True

# Which targets to process ([] means all)

just_targets = ['ngc4321']

# Which bands to use as internal references

ref_filt_miri = 'F770W'
ref_filt_nircam = 'F300M'

# Which bands to use as external comparisons

ext_comp_filt_miri = 'F770W'
ext_comp_filt_nircam = 'F300M'

# -----------------------------------------------------------------------------------
# Local directory structure
# -----------------------------------------------------------------------------------

# Locations

# ... of original data from the pipeline
my_input_root = '../orig_data/v0p7p3/'

# ... directory of comparison files (IRAC, WISE, etc.)
my_external_comp_dir = '../orig_data/background_comps/'

# ... of convolution kernels
my_kern_dir = '../orig_data/kernels/'

# ... of each output
my_output_root = '../working_data/processed_jwst/'
my_initial_matched_res_dir = my_output_root + 'initial_matched_res/'
my_anchored_dir = my_output_root + 'anchored/'
my_anchored_matched_res_dir = my_output_root + 'anchored_matched_res/'
my_plot_dir = my_output_root + 'background_plots/'
my_table_dir = my_output_root + 'background_tables/'

# First safe Gaussian and S/N price point from F2100W
ext_first_gauss = '_to_Gauss_0.850.fits'
label_first_gauss = 'atGauss0p85'
fwhm_first_gauss = 0.850

# List of Gaussian kernel targets for comparisons to external files

comp_gauss_dict = {
    'atGauss4' : 4.0,
    'atGauss7p5' : 7.5,
    'atGauss15' : 15,
}

comp_gauss_oversamp_dict = {
    'atGauss4' : 50,
    'atGauss7p5' : 200,
    'atGauss15' : 200,
}

# List of Gaussian kernel targets for output

output_gauss_dict = {
    'atGauss4' : 4.0,
    'atGauss7p5' : 7.5,
    'atGauss15' : 15,
}

output_gauss_oversamp_dict = {
    'atGauss4' : 50,
    'atGauss7p5' : 200,
    'atGauss15' : 200,
}

# -----------------------------------------------------------------------------------
# Directory structure
# -----------------------------------------------------------------------------------

# This could be replaced with TOML files and markup. It just defines
# the galaxy names, filters, and file locations for the original
# release.

image_key = Table.read('jwst_image_key.txt', format='ascii.csv',comment='#')
gal_names = np.unique(np.array(image_key['galaxy']))
filters = np.unique(np.array(image_key['filter']))
