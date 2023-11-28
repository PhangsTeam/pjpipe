======================
SingleTileDestripeStep
======================

**This step should be run on NIRCam only!**

Provides a number of methods to attempt to remove 1/f noise in NIRCam data just using the information in each tile
(as opposed to the :doc:`MultiTileDestripeStep <multi_tile_destripe>`, which runs on everything). If you have extended
emission in your data, set ``filter_diffuse=True``. We've seen good results with the default settings, but they can be
tuned to your liking. Can be run on both the output from level 1 (i.e. ``rate.fits`` files) which we recommend, or from
level 2 and beyond (i.e. ``cal.fits`` files). We recommend running straight after level 1 as the stripes should ideally
be corrected before flat-fielding.

If you have subarray observations, ``quadrants`` will automatically be set to ``False``, as the readout is different
in these modes.

---
API
---

.. autoclass:: pjpipe.SingleTileDestripeStep
    :members:
    :undoc-members:
    :noindex: