===========
ReleaseStep
===========

This step will flatten down the reprocessing and remove extraneous files for release. It also has the option to remove
generally unnecessary extensions from the various ``.fits`` files, which can significantly decrease the size of files
(particularly mosaics with large numbers of input tiles).

---
API
---

.. autoclass:: pjpipe.ReleaseStep
    :members:
    :undoc-members:
    :noindex:
