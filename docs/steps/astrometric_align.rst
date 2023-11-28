====================
AstrometricAlignStep
====================

This step provides absolute astrometric alignment to some external reference catalog, and back-propagates this to the
``crf.fits`` files. It allows for multiple iterations, to get e.g. a rough idea of the shift and then adding in rotation
to improve the solution. We can also inherit solutions from other bands, for e.g. long wavelength MIRI observations
where we find very few point sources so this kind of matching doesn't work. There is also an option for
cross-correlating the images to try and find a shift, but our testing finds this does not work particularly well and
simply inheriting the ``tweakreg`` solutions from a shorter wavelength band works better.

---
API
---

.. autoclass:: pjpipe.AstrometricAlignStep
    :members:
    :undoc-members:
    :noindex:
