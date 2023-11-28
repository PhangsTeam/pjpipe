=======
Lv2Step
=======

This is a light wrapper around the level 2 pipeline stage of the official pipeline.
For more details, see the `official documentation <https://jwst-pipeline.readthedocs.io/en/latest/jwst/pipeline/calwebb_image2.html#calwebb-image2>`_.

If you have background observations, you should ensure these are being correctly picked up by using ``bgr_check_type``
and ``bgr_background_name``.

---
API
---

.. autoclass:: pjpipe.Lv2Step
    :members:
    :undoc-members:
    :noindex:
