# jwst_scripts
PHANGS-JWST processing scripts

## Quickstart

The pipeline is designed to be run inside a python environment that can run the [JWST pipeline](https://jwst-pipeline.readthedocs.io/en/latest/).  
This Quickstart assumes a conda environment.

N.B. The pipeline is built to run with the latest GitHub version of the JWST pipeline. It may run in older versions,
but is likely to throw up error messages.

1. [Install and activate](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html) a `jwst-pipeline` conda environment.
2. Clone the `jwst_scripts` directory in to a `/path/`
3. Edit the `/path/config/local.toml` script to indicate where you want JWST processing to occur on your system.
4. Edit the `/path/config/config.toml` to reflect the galaxies you want to update.  The default will process NGC 1385.
5. Download the data from STScI.
```
python /path/download_phangs_jwst.py /path/config/config.toml
```
6a. Run the pipeline from the system command line:
```
python /path/run_jwst_reprocessing.py /path/config/config.toml
```
6b. If running inside an ipython shell, the script will load `/path/config/config.toml` automatically.


The resulting images will be stored in the working directory stored in your `working` directory defined in the `local.toml` file.

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

## General tools

* `check_archive_files` will check through downloaded files, to see if there are any issues and things need to be 
  redownloaded
* `compare_different_reprocess` will put out difference maps for two different reprocess versions, for regression 
  testing
* `get_wcs_adjust` will get WCS shifts and output in a format you can paste straight into a config file. It needs to run
  on data processed up to just before level 3, and will output a 'wcs_adjust.toml' file into the reprocess directory
* `psf_subtraction` has routines for PSF modelling/subtraction for saturated data
* `convert_con_to_coverage` will convert the CON file present in the release to a simpler map of the number of pixels
  contributing to the final mosaic

## PSF tools

* the PSF/ directory includes routines related to PSF creation and validation.
* 'make_kernels' in that directory creates circularized kernels from JWST or Gaussian PSFs, using the Aniano method.
* `generate_JWST_PSF_and_kernels` in that directory is a wrapper around `make_kernels`
*  `conv_with_kernel` in 'utils_jwst' can be used with the kernels 'make_kernels' (or they can be used by hand)

(These may be out of date and to be deprecated - they live in the home directory)
* `loop_kernel_creation` is an older version of material in the PSF directory
* `jwst_pypherise` uses pypher to produce PSF kernels to convert from JWST to JWST or JWST to gaussian/moffat

## PCA tools
* the `pca/` directory contains a number of robust PCA routines that we use for destriping data. They were ported to 
  Python by Elizabeth Watkins from IDL. These should reference Budavari+ 2009 (MNRAS 394, 1496–1502), and Wild+Hewett
  2005 (MNRAS 358, 1083-1099)

## Alignment files
* the `alignment/` directory contains .fits tables to provide absolute astrometric corrections

## Config files
* We use TOML files to simplify the interface between the processing parameters and the pipeline itself. The parameters
   we use for the PHANGS-JWST reduction are in the `config/` directory, but may need to be edited somewhat for other
   programs. The pipeline stage-specific parameters are given on the JWST pipeline readthedocs
   (https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/main.html#pipelines).

## Requirements

Beyond the requirements required by the official STScI pipeline, this pipeline also requires some packages not
installed by default:

* astroquery
* drizzlepac
* image-registration (install the GitHub version: pip install git+https://github.com/keflavich/image_registration.git)
* lmfit (only required for PSF modelling)
* numdifftools (only required for PSF modelling)
* pytest
* reproject
* tomli (not required for python>=3.11)
* tqdm
* webbpsf (only required for PSF modelling)

