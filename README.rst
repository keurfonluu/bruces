bruces
======

|License| |Stars| |Pyversions| |Version| |Downloads| |Code style: black| |Codacy Badge| |Codecov| |Build| |Docs| |DOI|

Inspired by `bruges <https://github.com/agilescientific/bruges>`__, **bruces** aims to be a collection of lightweight codes/tools for seismology with an emphasis on computational efficiency.

Installation
------------

The recommended way to install **bruces** and all its dependencies is through the Python Package Index:

.. code:: bash

   pip install bruces[full] --user

Otherwise, clone and extract the package, then run from the package location:

.. code:: bash

   pip install .[full] --user

To test the integrity of the installed package, check out this repository and run:

.. code:: bash

   pytest

Documentation
-------------

Refer to the online `documentation <https://keurfonluu.github.io/bruces/>`__ for detailed description of the API and examples.

Alternatively, the documentation can be built using `Sphinx <https://www.sphinx-doc.org/en/master/>`__:

.. code:: bash

   pip install -r doc/requirements.txt
   sphinx-build -b html doc/source doc/build

Example
-------

The following code snippet will decluster a catalog downloaded with `pycsep <https://github.com/SCECcode/pycsep>`__ using the nearest-neighbor method:

.. code-block:: python

   from datetime import datetime

   import csep
   import matplotlib.pyplot as plt

   import bruces

   # Download catalog using pycsep
   catalog = csep.query_comcat(
      start_time=datetime(2008, 1, 1),
      end_time=datetime(2018, 1, 1),
      min_magnitude=3.0,
      min_latitude=35.0,
      max_latitude=37.0,
      min_longitude=-99.5,
      max_longitude=-96.0,
   )

   # Decluster pycsep catalog
   cat = bruces.from_csep(catalog)
   eta_0 = cat.fit_cutoff_threshold()
   catd = cat.decluster(eta_0=eta_0)

   # Display declustering result
   fig, ax = plt.subplots(1, 2, figsize=(12, 6))
   cat.plot_time_space_distances(eta_0=eta_0, eta_0_diag=eta_0, ax=ax[0])
   catd.plot_time_space_distances(eta_0=eta_0, eta_0_diag=eta_0, ax=ax[1])

.. figure:: https://raw.githubusercontent.com/keurfonluu/bruces/4272457d2421697833514c5c08ad6b2ccf105748/.github/sample.svg
   :alt: sample
   :width: 100%
   :align: center

Contributing
------------

Please refer to the `Contributing
Guidelines <https://github.com/keurfonluu/bruces/blob/master/CONTRIBUTING.rst>`__ to see how you can help. This project is released with a `Code of Conduct <https://github.com/keurfonluu/bruces/blob/master/CODE_OF_CONDUCT.rst>`__ which you agree to abide by when contributing.

Notice
------

bruces Copyright (c) 2022, The Regents of the University of California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S. Dept. of Energy). All rights reserved.
If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual Property Office at `IPO@lbl.gov <mailto:IPO@lbl.gov>`__.

This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.

.. |License| image:: https://img.shields.io/badge/license-BSD--3--Clause-green
   :target: https://github.com/keurfonluu/bruces/blob/master/LICENSE

.. |Stars| image:: https://img.shields.io/github/stars/keurfonluu/bruces?logo=github
   :target: https://github.com/keurfonluu/bruces

.. |Pyversions| image:: https://img.shields.io/pypi/pyversions/bruces.svg?style=flat
   :target: https://pypi.org/pypi/bruces/

.. |Version| image:: https://img.shields.io/pypi/v/bruces.svg?style=flat
   :target: https://pypi.org/project/bruces

.. |Downloads| image:: https://pepy.tech/badge/bruces
   :target: https://pepy.tech/project/bruces

.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat
   :target: https://github.com/psf/black

.. |Codacy Badge| image:: https://img.shields.io/codacy/grade/27f1025983384885a3ed0f1089d3775e.svg?style=flat
   :target: https://www.codacy.com/gh/keurfonluu/bruces/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=keurfonluu/bruces&amp;utm_campaign=Badge_Grade

.. |Codecov| image:: https://img.shields.io/codecov/c/github/keurfonluu/bruces.svg?style=flat
   :target: https://codecov.io/gh/keurfonluu/bruces

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6422572.svg?style=flat
   :target: https://doi.org/10.5281/zenodo.6422572

.. |Build| image:: https://img.shields.io/github/workflow/status/keurfonluu/bruces/Python%20package
   :target: https://github.com/keurfonluu/bruces

.. |Docs| image:: https://img.shields.io/github/workflow/status/keurfonluu/bruces/Build%20documentation?label=docs
   :target: https://keurfonluu.github.io/bruces/