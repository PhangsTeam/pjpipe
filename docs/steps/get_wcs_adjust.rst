================
GetWCSAdjustStep
================

This step calculates shifts to get good relative alignment between tiles in a mosaic. Since longer wavelengths may not
have sources (and hence astrometry may fail catastrophically), we can instead defined a reference band to calculate a
shift per-visit (since JWST cycles between filters), and apply that shift to all tiles in that visit. Our testing shows
that taking a long NIRCam band that is ideally stellar dominated (i.e. F300M) works best for NIRCam, and a shorter
wavelength, non-PAH-dominated MIRI band (i.e. F1000W) is ideal for MIRI.

---
API
---

.. autoclass:: pjpipe.GetWCSAdjustStep
    :members:
    :undoc-members:
    :noindex:
