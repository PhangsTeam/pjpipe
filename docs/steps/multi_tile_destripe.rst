=====================
MultiTileDestripeStep
=====================

**This step should only be run on NIRCam data!**

This step uses information from all available tiles to correct any remaining 1/f noise. This is particularly important
if you've filtered out diffuse emission in :doc:`SingleTileDestripeStep <single_tile_destripe>`, as this will leave
large ripples untouched. This works by creating a mean image of the input tiles, and comparing each individual tile
to this mean. The hope is that any remaining 1/f noise will be averaged out, so we can use this to cleanly separate
source from instrumental noise.

If you have a small number of dithers, or at shorter NIRCam wavelengths, we suggest turning ``do_large_scale = True``
with a ``large_scale_filter_scale`` of ~200. This may need to be tweaked per-dataset. In this case, after the first pass we perform
the same average imaging, but smooth it perpendicular to the stripe direction, to hopefully remove any remaining
ripples. We then compare each tile to this smoothed, stacked image.

---
API
---

.. autoclass:: pjpipe.MultiTileDestripeStep
    :members:
    :undoc-members:
    :noindex: