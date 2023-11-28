############
PJPipe Steps
############

PJPipe has a number of steps. Here is a complete list, in approximately the
order they are expected to be run in (although there is some flexibility).
Each step has a ``do_step`` call that will run the step: ::

    from pjpipe import AStep

    step = AStep(*args)
    result = step.do_step()

However, we highly recommend running these are part of the integrated pipeline.

.. toctree::
    :maxdepth: 1
    :caption: Steps

    steps/pipeline.rst
    steps/download.rst
    steps/gaia_query.rst
    steps/move_raw_obs.rst
    steps/lv1.rst
    steps/single_tile_destripe.rst
    steps/lv2.rst
    steps/get_wcs_adjust.rst
    steps/apply_wcs_adjust.rst
    steps/lyot_separate.rst
    steps/lyot_mask.rst
    steps/level_match.rst
    steps/multi_tile_destripe.rst
    steps/psf_model.rst
    steps/lv3.rst
    steps/astrometric_catalog.rst
    steps/astrometric_align.rst
    steps/mosaic_individual_fields.rst
    steps/anchoring.rst
    steps/psf_matching.rst
    steps/release.rst
    steps/regress_against_previous.rst
