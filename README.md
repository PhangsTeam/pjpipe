# jwst_scripts
PHANGS-JWST processing scripts

## Query/Download tools

* `archive_download` is a generic wrapper around Astroquery MAST query/download
* `download_phangs_jwst` can be used to download the PHANGS-JWST data into a reasonable directory structure

## NIRCAM tools

* `nircam_destriping` contains various routines for destriping NIRCAM data
* `run_nircam_destriping` is a wrapper to run the destriping for select NIRCAM frames
* `jwst_nircam_reprocess` used to destripe and reprocess level3 NIRCAM data
* `run_nircam_reprocessing` wraps around `jwst_nircam_reprocess`
* `log_jwst_nircam_reprocessing` logs `run_nircam_reprocessing` to a file

## MIRI tools

* `jwst_miri_reprocess` is used to associate OFF and science MIRI exposures and 
to reprocess the level1 (if necessary), level2 and level3 MIRI data reduction 
* `miri_destriping` contains routines for destriping MIRI data
* `run_miri_destriping` wraps around `miri_destriping`

## Data homogenisation tools

* `make_jwst_kernels` creates convolution kernels from JWST or Gaussian PSFs, using the Aniano method. For now at least,
  these kernels are circularised
* `loop_kernel_creation` provides an example wrapper around `make_jwst_kernels` 
* `jwst_pypherise` uses pypher to produce PSF kernels to convert from JWST to JWST or JWST to gaussian/moffat

## PCA tools
* the `pca/` directory contains a number of robust PCA routines that we use for destriping data. They were ported to 
  Python by Elizabeth Watkins from IDL. These should reference Budavari+ 2009 (MNRAS 394, 1496–1502), and Wild+Hewett
  2005 (MNRAS 358, 1083-1099)
