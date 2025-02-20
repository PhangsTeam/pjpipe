1.2.1 (Unreleased)
==================

- Fix for crash if trying to anchor without first aligning
- Fix for crash if trying to PSF match without some combination of anchoring, aligning
- Change how the level 3 step handles models for newer JWST versions

1.2.0 (2024-09-03)
==================

- Removed raised warning in ``multi_tile_destripe_step`` when input images have a variety of
  rotations
- Added ``combine_nircam_short`` option in ``level_match_step``, which will match levels
  between the four NIRCam short imaging chips before doing matching between mosaic tiles
- ``lv2_step`` will now propagate through individual exposure offset times from backgrounds,
  which is necessary for selecting a reference image in ``level_match_step``
- Added ``fit_type`` option to ``level_match_step`` for every option, which allows for
  plane fitting in the level matching in a modular way
- ``level_match_step`` fitting method has been changed to resemble the iterative Montage method,
  which is necessary for plane fitting
- ``reproject_image`` in ``utils`` can now also reproject error arrays
- ``make_stacked_image`` in ``utils`` can now also reproject the error and readnoise maps
- Exposed ``auto_rotate`` in ``make_stacked_image`` in ``utils``
- Added prefilter option to ``get_wcs_adjust``, which uses constrained diffusion to remove large-scale structure
- Added ``recombine_lyot`` option to ``level_match_step``, which allows for recombining
  the lyot coronagraph into the main MIRI science chip before matching between mosaic tiles
- Tidied up ``level_match_step`` to make more modular and functional
- Fix bug in ``astrometric_align_step`` if first pass of alignment succeeds but second
  fails
- Add cross-correlation option to ``get_wcs_adjust_step``, which uses ``spacepylot``
- Add control over reproject functions, can be ``reproject_interp`` (default), ``reproject_exact``,
  or ``reproject_adaptive``. For MIRI, ``reproject_exact`` may work better. This applies to ``anchoring_step``,
  ``astrometric_align_step``, ``level_match_step``, ``multi_tile_destripe_step``, ``psf_matching_step``, and
  ``release_step``
- Fix bug for lv1 hanging on newer Macbooks
- Fix bug for parameters with 'pix' in getting picked up like numbers of pixels
- Updated PHANGS Cy3 config
- Include links to Francesco Belfiore's kernel generation repository in the docs
- Updated various package requirements
- Updated indexing to align with new ``reproject`` updates

1.1.0 (2024-03-04)
==================

- If science lv1 data already exists, just copy it over for background data to save processing time
- Include BMAJ in ``psf_matching_step``
- Auto rotate the WCS in ``level_match_step``
- Fix SIP approximation issues in ``apply_wcs_adjust_step``
- ``astrometric_catalog_step`` can now use either DAOStarFinder or IRAFStarFinder. Observatory recommendations seem
    to be IRAFStarFinder in general
- Removed factor of 2 for FWHM in lv3 tweakreg-related steps to more closely match observatory recommendations
- Add lower version pins to project requirements to avoid incompatibilities
- Fixed bug in ``astrometric_align_step`` where file might already be closed before saving
- Updated PHANGS Cycle 1 config to v1p1 (last couple of observations, and a few small changes to the config)
- Fix bug with ``process_bgr_like_science`` in ``lv3_step``
- Added 'filename' option for background association
- Fix in ``parse_fits_to_table`` when there is not a defined observation label
- Point Zenodo DOI to "concept" DOI, rather than latest
- Cleaner labels in ``anchoring_step``
- Add acknowledgment for PCA (thanks Liz!)
- Added CITATION.cff

1.0.4 (2024-01-04)
==================

- Fix del in ``get_wcs_adjust`` if we skip alignment
- Made plots more consistent and "publication ready" across the board
- Fix bug in ``anchoring step``
- Rename ``anchor_to_external_step.py`` to ``anchor_step.py`` for consistency
- Fix "too many plots open" warning in ``anchoring_step``
- Include useful pjpipe version info in files
- Save anchoring background in metadata properly in ``anchoring_step``
- Add Zenodo badge

1.0.2 (2023-12-01)
==================

- Fix packaging to minimise file size
- Error in ``pyproject.toml``

1.0.0 (2023-12-01)
==================

- Added documentation
- Ensure ``multi_tile_destripe_step`` is properly imported in the pipeline
- Remove an error in ``multi_tile_destripe_step`` stripes in stacked image don't align with any particular
  axis of the array
- Fix crash where quadrants are turned off in median filtering in ``single_tile_destripe_step``
- Fix potential memory leaks in ``get_wcs_adjust_step`` and ``apply_wcs_adjust_step``
- Diagnostic plots are compressed by default in ``release_step``

0.10.1 (2023-11-20)
===================

- Can specify CRDS context at the pipeline level
- median_filter is now the default in ``single_tile_destripe_step``
- Added fallback for median_filter in ``single_tile_destripe_step`` when too much data is masked
- Remove mask option in median_filter in ``single_tile_destripe_step``, since it's always used
  anyway
- Changed up how ``do_large_scale`` works in ``multi_tile_destripe_step``,
  which seems significantly improved and simplified
- Added fallback for in ``multi_tile_destripe_step`` when too much data is masked in quadrants
- Changed how tweakreg grouping is done in ``get_wcs_adjust_step`` and ``lv3_step`` to account
  for code changes in the pipeline
- Added option to decouple the short NIRCam chips for tweakreg in ``lv3_step``
- Added option to move the various diagnostic plots in ``release_step``
- ``remove_bloat`` is now False by default in ``release step``, to maintain datamodels compatibility
- ``remove_bloat`` also applies to PSF matched files in ``release_step``
  
0.10.0 (2023-11-14)
===================

- Added PSF modelling routines. These are currently very preliminary, but at least exist
- Allow for external, absolute catalog in ``get_wcs_adjust_step``
- Add PSF matching routines (``psf_matching``)
- Add anchoring routines (``anchoring``)
- Include useful outputs from these in the ``release_step``
- If not grouping dithers in ``get_wcs_adjust_step``, respect that in how the transforms are
  written out
- ``single_tile_destripe_step`` can now run on rate files (pre-flat fielding)
- Bugfixing in ``single_tile_destripe_step``
- Decoupled horizontal/vertical destriping methods in ``single_tile_destripe_step``, since the
  noise properties are distinct in these two axes
- Add ``smooth`` option to ``single_tile_destripe_step``, based on Dan Coe's smooth1overf
  algorithm
- Added control over how values are extended beyond array edge for filtering in ``single_tile_destripe_step``
- Lots of bugfixing in ``multi_tile_destripe_step``
- ``multi_tile_destripe_step`` can un-flat before correcting
- Level between amplifiers in ``multi_tile_destripe_step``
- Make vertical stripe subtraction optional in ``multi_tile_destripe_step``
- Added median option to ``multi_tile_destripe_step`` to do a median rather than mean image
- Added iterative option to ``multi_tile_destripe_step`` that will keep things going until
  sigma-based convergence
- Added sigma-clip median option for creating stacked images
- ``do_large_scale`` now works completely differently in ``multi_tile_destripe_step``, instead
  attempting to clean up the average image
- Added support for different ``do_large_scale`` methods in ``multi_tile_destripe_step``,
  which may work better in certain situations
- Added a median filter ``do_large_scale`` method, which may be more robust than the boxcar. THIS
  IS NOW THE DEFAULT
- Added a sigma-clipped ``do_large_scale`` method, as should be optimal in observations that aren't
  full of emission
- Added a smooth convolution ``do_large_scale`` method, based on Dan Coe's smooth1overf algorithm
- Added control over how values are extended beyond array edge for ``do_large_scale``
- Added option in ``lv3_step`` to degroup dithers for tweakreg
- Added option in ``lv3_step`` to degroup NIRCam modules, since the WCS is currently inconsistent
  between the two
- Fixed crash in ``lv3_step`` if one of the group/degroup parameters is not defined
- Be smarter about keeping track of exposure numbers in ``lv3_step``
- ``regress_against_previous`` will now search for files in priority order, for fallback between versions
- Make sure backgrounds are included in label for ``regress_against_previous``
- f-string fixes

0.9.2 (2023-09-18)
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