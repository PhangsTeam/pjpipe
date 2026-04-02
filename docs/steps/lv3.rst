=======
Lv3Step
=======

This is a light wrapper around the level 3 pipeline stage of the official pipeline.
For more details, see the `official documentation <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_image3.html>`_.

If you have background observations, you should ensure these are being correctly picked up by using ``bgr_check_type``
and ``bgr_background_name``.

After the main pipeline run, individual exposures can optionally be drizzled onto the
mosaic WCS (boolean ``do_drizzle`` flag) and then blotted back to a reference detector frame
(boolean ``do_blot`` flag). This is useful for per-exposure comparisons in pixel space. Resample
parameters are read from ``jwst_parameters["resample"]``; blotting behaviour is
controlled by ``blot_ref_index`` and ``blot_fillval``. In your config file:

Example 1: Just drizzle the individual exposures onto the mosaic WCS. No blotting.
.. code-block:: toml

   [parameters.lv3]
   do_drizzle = true
   do_blot = false

Example 2: Drizzle the individual exposures onto the mosaic WCS and then blot back to the reference detector frame
.. code-block:: toml

   [parameters.lv3]
   # You can also set do_drizzle = true here, but not necessary.
   # You can not set do_blot = true while setting do_drizzle = false (you'll get an error)
   do_blot = true
   blot_ref_index = 0
   blot_fillval = np.nan

---
API
---

.. autoclass:: pjpipe.Lv3Step
    :members:
    :undoc-members:
    :noindex:
