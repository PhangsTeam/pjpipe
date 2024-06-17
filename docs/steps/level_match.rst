==============
LevelMatchStep
==============

This step matches relative levels between tiles, using an algorithm very similar to
`Montage <http://montage.ipac.caltech.edu/>`_. We reproject tiles to a common astrometric grid, and then for each
overlapping pair find the median per-pixel difference, and minimize for that. We do this in a two (or three)-pass
process, firstly for dithers within a mosaic tile, then optionally for main science chip and lyot coronagraph,
optionally between the four NIRCam short chips, and then for all mosaic tiles (creating stacked images). This maximises
overlaps between mosaic tiles and produces better results from our testing.

For MIRI imaging, there is an optional ``recombine_lyot`` parameter. If you've used the ``lyot_separate`` step then
this will match the coronagraph and main science chip and recombine them before any potential matching between
different tiles in the mosaic.

For NIRCam short imaging, there is an optional ``combine_nircam_short`` parameter. This will match levels between the
four NIRCam short chips and then will treat as one single image going into the final mosaic level matching stage.

For each of these three stages, you can match either with a simple constant offset ("level") or with a plane
("level+match"). This can be controlled separately for each stage (``fit_type_dithers``, ``fit_type_recombine_lyot``,
``fit_type_combine_nircam_short``, ``fit_type_mosaic_tiles``). By default, everything will just fit a constant offset.

When matching between mosaic tiles, we select a reference image that we deem to be the most flat, and will adjust any
corrections relative to that. If you are using observations with dedicated backgrounds (as defined by
:doc:`Lv2Step <lv2>`, then it will select the image closest in time to the backgrounds. If not, it will use the image
with the most overlapping image pairs.

N.B. This should be run once you have good local astrometry, and before you do
:doc:`MultiTileDestripeStep <multi_tile_destripe>`.

---
API
---

.. autoclass:: pjpipe.LevelMatchStep
    :members:
    :undoc-members:
    :noindex: