================
LyotSeparateStep
================

**This step should only be used on MIRI data!**

This step splits up the MIRI image into the Lyot coronagraph and the main science chip. This means we can match the
different backgrounds in each in a consistent way, and include the coronagraph in the final images.

N.B. only use one of either this step or :doc:`LyotMaskStep <lyot_mask>`.

---
API
---

.. autoclass:: pjpipe.LyotSeparateStep
    :members:
    :undoc-members:
    :noindex:
