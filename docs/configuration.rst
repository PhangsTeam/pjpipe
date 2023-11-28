###################
Configuration Files
###################

The pipeline is primarily interfaced with using config files. These are .toml,
and as such are relatively human-readable. We separate out parameters that
control the overall pipeline processing (config files) and ones that are
specific to the system directory layout (local files).

At the very least, you should determine a list of targets: ::

    targets = [
       'ic5332',
       'ngc0628',
       'ngc1087',
       'etc',
    ]

a version: ::

    version = 'v0p8p2'

a list of bands: ::

    bands = ['F300M']

and some steps: ::

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

Note that in the steps here, you can separate things out per-instrument.
For instance, destriping only runs on NIRCam images, whilst anything to
do with the lyot coronagraph only applies to MIRI.

There is also the option to separately image background observations. For
this, append ``_bgr`` to a band: ::

    bands = [
        'F2100W',
        'F2100W_bgr',
    ]

and you can also run steps separately depending on whether the observations
are background or not: ::

    steps = [
        'lyot_separate.miri.sci',
        'lyot_mask.miri.bgr',
    ]

This config file controls the parameters for each step. You can edit
things like so: ::

    [parameters.download]

    prop_id = '2107'
    product_type = [
        'SCIENCE',
    ]
    calib_level = [
        1,
    ]

which will download data from Program ID 2107 (the PHANGS Cycle 1 Treasury),
and only level 1 (uncal) files. For more examples, we include examples at the
end of this page. For any parameters passed to the JWST pipeline itself,
these should be nested as ``jwst_parameters``, e.g.: ::

    [parameters.lv1]

    jwst_parameters.save_results = true
    jwst_parameters.ramp_fit.suppress_one_group = false
    jwst_parameters.refpix.use_side_ref_pixels = true


The ``local.toml`` file simply defines where things will be saved. For example, ::

    crds_path = '/data/beegfs/astro-storage/groups/schinnerer/williams/crds/'
    raw_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_raw/archive_20230711/'
    reprocess_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_phangs_reprocessed/'
    alignment_dir = '/data/beegfs/astro-storage/groups/schinnerer/williams/jwst_scripts/examples/2107/alignment/'
    processors = 20

This should be edited to match your system layout.

========
Examples
========

Here we provide some example configuration files, for inspiration

--------------
PHANGS Cycle 1
--------------

``config.toml``:

.. literalinclude:: ../config/2107/config.toml

``local.toml``:

.. literalinclude:: ../config/2107/astronode.toml

--------------
PHANGS Cycle 2
--------------

``config.toml``:

.. literalinclude:: ../config/3707/config.toml
