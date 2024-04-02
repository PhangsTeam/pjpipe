==============
LevelMatchStep
==============

This step matches relative levels between tiles, using an algorithm very similar to
`Montage <http://montage.ipac.caltech.edu/>`_. We reproject tiles to a common astrometric grid, and then for each
overlapping pair find the median per-pixel difference, and minimize for that. We do this in a two-pass process, firstly
for dithers within a mosaic tile, and then for all mosaic tiles (creating stacked images). This maximises overlaps
between mosaic tiles and produces better results from our testing.

For MIRI imaging, there is an optional ``recombine_lyot`` parameter. If you've used the ``lyot_separate`` step then
this will match the coronagraph and main science chip and recombine them before any potential matching between
different tiles in the mosaic.

N.B. This should be run once you have good local astrometry, and before you do
:doc:`MultiTileDestripeStep <multi_tile_destripe>`.

---
API
---

.. autoclass:: pjpipe.LevelMatchStep
    :members:
    :undoc-members:
    :noindex: