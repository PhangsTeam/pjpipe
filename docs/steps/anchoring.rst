=============
AnchoringStep
=============

This step will anchor images to an external reference level, and then will internally match other bands to this
externally matched band. You will need to provide convolution kernels and images.

If you want to generate a kernel that can feed directly into this step, please see
`this repository <https://github.com/francbelf/jwst_kernels>`_.

---
API
---

.. autoclass:: pjpipe.AnchoringStep
    :members:
    :undoc-members:
    :noindex:
