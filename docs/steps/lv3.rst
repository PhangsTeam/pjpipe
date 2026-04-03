=======
Lv3Step
=======

This is a light wrapper around the level 3 pipeline stage of the official pipeline.
For more details, see the `official documentation <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_image3.html>`_.

If you have background observations, you should ensure these are being correctly picked up by using ``bgr_check_type``
and ``bgr_background_name``.

After the main pipeline run, two optional post-processing steps are available.
These are independent of each other and can be enabled separately:

- ``do_drizzle``: Drizzle individual exposures onto the mosaic WCS, producing
  one ``*_i2d_single.fits`` file per exposure. Resample parameters are read
  from ``jwst_parameters["resample"]``.
- ``do_blot``: Blot the final i2d mosaic back to each exposure's detector
  frame, producing one ``*_i2d_blot.fits`` file per exposure. The fill value
  for pixels outside the blotted footprint is controlled by ``blot_fillval``.

**Example 1** -- Drizzle individual exposures onto the mosaic WCS only:

.. code-block:: toml

   [parameters.lv3]
   do_drizzle = true

**Example 2** -- Blot the mosaic to every exposure's detector frame only:

.. code-block:: toml

   [parameters.lv3]
   do_blot = true
   blot_fillval = "nan"

**Example 3** -- Both drizzle and blot:

.. code-block:: toml

   [parameters.lv3]
   do_drizzle = true
   do_blot = true

---
API
---

.. autoclass:: pjpipe.Lv3Step
    :members:
    :undoc-members:
    :noindex:
