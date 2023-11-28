============
DownloadStep
============

This step automates downloading from MAST, sorting things into folders
with the layout the pipeline will expect going forwards. You should pass
at least a target or a proposal ID, and for proprietary data you should
set login to ``True``, at which point it will ask for a login API key.
For large targets or those with backgrounds that are far away, you may
need to bump up the radius parameter.

---
API
---

.. autoclass:: pjpipe.DownloadStep
    :members:
    :undoc-members:
    :noindex:
