===============
PersistenceStep
===============

As MIRI does not currently have a persistence correction step, we implement one
based on `this paper <https://arxiv.org/pdf/2512.15477>`_. For each observation,
we calculate a percentage flux correction (in detector pixel space) based on previous
observations.

This step also contains options to include a contribution from previous observation
groups (rather than just the immediate dithers around the observation being considered),
as well as from other bands.

N.B. This step is currently undergoing testing, so a definitive recommendation on the settings
is not currently available

---
API
---

.. autoclass:: pjpipe.PersistenceStep
    :members:
    :undoc-members:
    :noindex:
