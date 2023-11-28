============
LyotMaskStep
============

**This step should only be used on MIRI data!**

This step will mask the Lyot coronagraph as 'non-science' data, so it won't be included in the final mosaic.

N.B. only use one of either this step or :doc:`LyotSeparateStep <lyot_separate>`.

---
API
---

.. autoclass:: pjpipe.LyotMaskStep
    :members:
    :undoc-members:
    :noindex:
