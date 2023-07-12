# These are user-specific options that set what
# generate_JWST_PSF_and_kernels does. Moved to a separate config-style
# file to make the git management easier.

# Control flow

do_kern_to_f2100w = False
do_kern_to_f360m = False
do_kern_to_copt = True
do_kern_to_gauss = False

# output directory where you want the JWST PSFs to be saved (relative for me to PSF/.)
output_dir_psf = 'PSF/'  # in PSF/PSF/ relative to the repo root
output_dir_kernels = '../../orig_data/kernels/'

# list of the PHANGS-JWST filters to consider, others can be added if necessary

nircam_psf_list = [
    'F200W',
    'F300M',
    'F335M',
    'F360M',
]

miri_psf_list = [
    'F770W',
    'F1000W',
    'F1130W',
    'F2100W',
]

# Gaussian resolution list

gauss_res_list = [
    0.75,0.8,0.85,0.9,0.95,1.0
]

# list of targets for which to attempt to build kernels to match the
# MUSE copt resolution

copt_targets = ['NGC0628', 'NGC1087', 'NGC1300', 'NGC1365',
                'NGC1385', 'NGC1433', 'NGC1512', 'NGC1566',
                'NGC1672', 'NGC3627', 'NGC4303', 'NGC4321',
                #'NGC4535',
                'NGC5068', 'NGC7496', 'IC5332']
