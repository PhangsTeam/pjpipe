# PJPipe

[![](https://img.shields.io/pypi/v/pjpipe.svg?label=PyPI&style=flat-square)](https://pypi.org/pypi/pjpipe/)
[![](https://img.shields.io/pypi/pyversions/pjpipe.svg?label=Python&color=yellow&style=flat-square)](https://pypi.org/pypi/pjpipe/)
[![Docs](https://readthedocs.org/projects/pjpipe/badge/?version=latest&style=flat-square)](https://pjpipe.readthedocs.io/en/latest/)
[![Actions](https://img.shields.io/github/actions/workflow/status/phangsTeam/pjpipe/build_test.yaml?branch=main&style=flat-square)](https://github.com/phangsTeam/pjpipe/actions)
[![License](https://img.shields.io/badge/license-GNUv3-blue.svg?label=License&style=flat-square)](LICENSE)

![](docs/images/pjpipe_logo.jpg)

**Note that this pipeline requires Python 3.9 or above**

PJPipe (the PHANGS-JWST-Pipeline) is a wrapper around the official 
[STScI JWST Pipeline](https://jwst-pipeline.readthedocs.io/en/latest) 
for imaging data (not spectroscopy), with edits specific to the reduction 
of large mosaics and nearby galaxies with extended, diffuse emission.

Beyond the standard pipeline, PJPipe offers options for 

* NIRCam destriping
* Dealing with the MIRI coronagraph
* Background matching
* Absolute astrometric correction

Alongside this, PJPipe is also highly parallelised for speed, and provides
a simple, high-level interface via configuration files.

If you make use of PJPipe in your work, please cite the PHANGS-JWST survey 
papers (Lee et al., 2022; Williams et al., in prep.), and do not hesitate to
get in touch for help! The `/config` directory on the 
[GitHub repository](https://github.com/phangsTeam/pjpipe) has examples, 
but different datasets may need more specific tailoring. You can open an 
[issue](https://github.com/PhangsTeam/pjpipe/issues) if you run into problems.

## Installation

The easiest way to install PJPipe is via pip: 

```bash
pip install pjpipe
```

## Setting up config files

The pipeline is primarily interfaced with using config files. These are .toml,
and as such are relatively human-readable. We separate out parameters that
control the overall pipeline processing (config files) and ones that are 
specific to the system directory layout (local files).

At the very least, you should determine a list of targets:
```toml
targets = [
    'ic5332',
    'ngc0628',
    'ngc1087',
    'etc',
]
```
a version:
```toml
version = 'v0p8p2'
```
a list of bands
```toml
bands = ['F300M']
```
and some steps
```toml
steps = [
    'download',
    'lv1',
    'lv2',
    'single_tile_destripe.nircam',
    'get_wcs_adjust',
    'apply_wcs_adjust',
    'lyot_separate.miri',
    'multi_tile_destripe.nircam',
    'level_match',
    'lv3',
    'astrometric_catalog.miri',
    'astrometric_align',
    'release',
    'regress_against_previous',
]
```
Note that in the steps here, you can separate things out per-instrument.
For instance, destriping only runs on NIRCam images, whilst anything to
do with the lyot coronagraph only applies to MIRI.

There is also the option to separately image background observations. For 
this, append `_bgr` to a band:
```toml
bands = [
    'F2100W',
    'F2100W_bgr',
]
```
and you can also run steps separately depending on whether the observations
are background or not:
```toml
steps = [
    'lyot_separate.miri.sci',
    'lyot_mask.miri.bgr',
]
```

This config file controls the parameters for each step. You can edit 
things like so:
```toml
[parameters.download]

prop_id = '2107'
product_type = [
    'SCIENCE',
]
calib_level = [
    1,
]
```
which will download data from Program ID 2107 (the PHANGS Cycle 1 Treasury),
and only level 1 (uncal) files. For more examples, we suggest looking in the 
`config/` directory. For any parameters passed to the JWST pipeline itself,
these should be nested as `jwst_parameters`, e.g.:
```toml
[parameters.lv1]

jwst_parameters.save_results = true
jwst_parameters.ramp_fit.suppress_one_group = false
jwst_parameters.refpix.use_side_ref_pixels = true
```

The `local.toml` file simply defines where things will be saved. For example,
```toml
crds_path = '/data/beegfs/astro-storage/groups/schinnerer/williams/crds/'
raw_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_raw/archive_20230711/'
reprocess_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_phangs_reprocessed/'
alignment_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_scripts/examples/2107/alignment/'
processors = 20
```
This should be edited to match your system layout.


## Running the Pipeline

After you have your config files set up (see the `/config` directory for some examples),
you can run the pipeline end-to-end with just a few lines:

```python
from pjpipe import PJPipeline

config_file = '/path/to/config.toml'
local_file = '/path/to/local.toml'

pjp = PJPipeline(config_file=config_file,
                 local_file=local_file,
                 )
pjp.do_pipeline()
```

Then just sit back and enjoy the heavy lifting being done.

### Downloading Reference Files

If this is your first time running anything JWST related, errors can
occur because the pipeline expects some reference files. To fix this,
after setting CRDS parameters run
```python
import os

os.system('crds sync --jwst')
```
and this will pull the minimum relevant files for you.

### Optional Arguments

Each step is highly configurable to the end user, although the defaults should be 
sensible in many use cases. To see all steps, you can simply do:
```python
import pjpipe

pjpipe.list_steps()
```
To see the possible arguments for each step, you can do:
```python
from pjpipe import DownloadStep

help(DownloadStep)
```

This will list the optional arguments you can put into the config file, 
and will be passed to the step. This is not necessarily true for each
JWST pipeline step, for those we recommend looking at 
[the online docs](https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/main.html#pipelines)

### Credits

PJPipe has been developed by the [PHANGS Team](phangs.org), with major contributions from:

* Thomas Williams (University of Oxford)
* Oleg Egorov (Universität Heidelberg)
* Erik Rosolowsky (University of Alberta)
* Francesco Belfiore (INAF)
* Jessica Sutter (UCSD)
* David Thilker (JHU)
* Adam Leroy (OSU)
