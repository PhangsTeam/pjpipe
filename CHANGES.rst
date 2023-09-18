0.9.2 (Unreleased)
==================

- Allow multiple options for e.g. proposal ID in ``download_step``
- Fix potential error with file validation in ``download_step``
- ``move_raw_obs_step`` is smarter about missing filters
- ``lv1_step`` is smarter about grouping dithers
- Fix plotting error if quadrants=False and using median filter in ``single_tile_destripe_step``
- Fix potential subarray issues with ``lyot_separate_step``/``lyot_mask_step``
- ``do_large_scale`` defaults to False in ``multi_tile_destripe_step``
- Much improved diagnostic plots in ``multi_tile_destripe_step``
- Catch errors in ``level_match_step`` where all data might be NaN
- Rename ``do_vertical_subtraction`` in config files
- Caught some typos from lazy copy/pasting docstrings
- Additions and updates for #2130 (Local Group) and #3707 (Cy2 Treasury)

0.9.1 (2023-09-04)
==================

- If not supplied, will default to running on all CPUs, not 1
- Include option for producing background images
- Include Gaia query for astrometric catalogs (``gaia_query_step``)
- Include option to produce mosaics for each individual field (``mosaic_individual_fields_step``)
- Parallelise up the download integrity verification in ``download_step``
- Catch warnings as errors in integrity verification in ``download_step``
- Include array information when creating asn files to ensure we don't associate backgrounds incorrectly
- Change naming system for lyot separate to ensure compatibility with later steps
- ``get_wcs_step`` now sorts shifts to be more human-readable
- Add local background subtraction to ``level_match_step``, which may help for mosaics without overlaps
- ``release_step`` now takes the lv3 directory as an argument, rather than parsing any progress dictionaries
- ``release_step`` will now also move any individual field mosaics

0.9 (2023-07-25)
================

- Modular refactor
- Include subtracted backgrounds in release