==========
PJPipeline
==========

This is the overall pipeline class. Assuming you have your config files set up correctly,
then you can run through the full pipeline in just a couple of line: ::

    import pjpipe

    pjp = pjpipe.PJPipeline(
        config_file=config_file,
        local_file=local_file,
    )
    pjp.do_pipeline()

---
API
---

.. autoclass:: pjpipe.PJPipeline
    :members:
    :undoc-members:
    :noindex:
