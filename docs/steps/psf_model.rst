============
PSFModelStep
============

**This should be considered pre-alpha, we have not got it working well yet!**

This step finds saturated sources in each individual tile and attempts to either 'paint in' the saturation via PSF
fitting or subtracting off the PSF to remove the wings that can extend over large portions of the image. Unfortunately,
the PSF models are not at the level required for good subtraction currently, and we have no way of dealing with the
`brighter-fatter effect <https://arxiv.org/abs/2303.13517>`_, so this step produces weird looking results.

---
API
---

.. autoclass:: pjpipe.PSFModelStep
    :members:
    :undoc-members:
    :noindex:
