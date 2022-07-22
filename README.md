# jwst_scripts
PHANGS-JWST processing scripts

## Query/Download tools

* `archive_download` is a generic wrapper around Astroquery MAST query/download
* `download_phangs_jwst` can be used to download the PHANGS-JWST data into a reasonable directory structure

## NIRCAM tools

* `nircam_destriping` contains various routines for destriping NIRCAM data
* `run_nircam_destriping` is a wrapper to run the destriping for one NIRCAM frame

## MIRI tools

* `jwst_miri_reprocess` is used to associate OFF and science MIRI exposures and 
to reprocess the level2 and level3 MIRI data reduction 

## Data homogenisation tools

* `jwst_pypherise` uses pypher to produce PSF kernels to convert from JWST to JWST or JWST to gaussian/moffat
