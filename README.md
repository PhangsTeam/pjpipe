# jwst_scripts
PHANGS-JWST processing scripts

## Query/Download tools

* `archive_download` is a generic wrapper around Astroquery MAST query/download
* `download_phangs_jwst` can be used to download the PHANGS-JWST data into a reasonable directory structure

## Reduction tools

* `jwst_reprocess` is used to reprocess both NIRCAM and MIRI data
* `run_jwst_reprocessing` wraps around `jwst_reprocess`
* `log_jwst_reprocessing` logs `run_jwst_reprocessing` to a file
* `prepare_release` flattens the file structure into something more manageable

## NIRCAM tools

* `nircam_destriping` contains various routines for destriping NIRCAM data
* `run_nircam_destriping` is a wrapper to run the destriping for select NIRCAM frames

## MIRI tools

* `miri_destriping` contains routines for destriping MIRI data
* `run_miri_destriping` wraps around `miri_destriping`

## Data homogenisation tools

* `make_kernels` creates convolution kernels from JWST or Gaussian PSFs, using the Aniano method. For now at least, these kernels are circularised
* `loop_kernel_creation` provides an example wrapper around `make_jwst_kernels` 
* `generate_JWST_PSF_and_kernels` provides an example wrapper around `make_jwst_kernels`  which generates the PSF and kernels inside the `PSF/` directory
*  `convolve_data` uses the kernels from `generate_JWST_PSF_and_kernels` to make convolved version of the fully reduced NIRCam and MIRI frames
* `jwst_pypherise` uses pypher to produce PSF kernels to convert from JWST to JWST or JWST to gaussian/moffat

## PCA tools
* the `pca/` directory contains a number of robust PCA routines that we use for destriping data. They were ported to 
  Python by Elizabeth Watkins from IDL. These should reference Budavari+ 2009 (MNRAS 394, 1496–1502), and Wild+Hewett
  2005 (MNRAS 358, 1083-1099)

## Alignment files
* the `alignment/` directory contains .fits tables to provide absolute astrometric corrections
