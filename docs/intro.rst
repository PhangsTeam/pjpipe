######
PJPipe
######

.. image:: https://img.shields.io/pypi/v/pjpipe.svg?label=PyPI&style=flat-square
    :target: https://pypi.org/pypi/pjpipe/
.. image:: https://img.shields.io/pypi/pyversions/pjpipe.svg?label=Python&color=yellow&style=flat-square
    :target: https://pypi.org/pypi/pjpipe/
.. image:: https://img.shields.io/github/actions/workflow/status/phangsTeam/pjpipe/build_test.yaml?branch=main&style=flat-square
    :target: https://github.com/phangsTeam/pjpipe/actions
.. image:: https://readthedocs.org/projects/pjpipe/badge/?version=latest&style=flat-square
   :target: https://pjpipe.readthedocs.io/en/latest/
.. image:: https://img.shields.io/badge/license-GNUv3-blue.svg?label=License&style=flat-square

**Note that this pipeline requires Python 3.9 or above**

PJPipe (the PHANGS-JWST-Pipeline) is a wrapper around the official
`STScI JWST Pipeline <https://jwst-pipeline.readthedocs.io/en/latest>`_,
for imaging data (not spectroscopy) with edits specific to the reduction of
large mosaics and nearby galaxies with extended, diffuse emission.

Beyond the standard pipeline, PJPipe offers options for

* NIRCam destriping
* Dealing with the MIRI coronagraph
* Background matching
* Absolute astrometric correction

Alongside this, PJPipe is also highly parallelised for speed, and provides
a simple, high-level interface via configuration files. The
:doc:`configuration page <configuration>` has configuration examples to get
you imaging your data, but different datasets may need more specific tailoring.

If you make use of PJPipe in your work, please cite the PHANGS-JWST survey
papers (Lee et al., 2022; Williams et al., in prep.), and do not hesitate to
get in touch for help! You can open an
`issue <https://github.com/PhangsTeam/pjpipe/issues>`_ if you run into problems.
