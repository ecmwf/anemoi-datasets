.. _installing:

############
 Installing
############

****************
 Python Version
****************

-  Python (> 3.9)

We require at least Python 3.9.

**************
 Installation
**************

Environments
============

We currently do not provide a conda build of anemoi-datasets, so the
suggested installation is through Python virtual environments.

For Linux, the process to make and use a venv is as follows:

.. code:: bash

   python -m venv /path/to/my/venv
   source /path/to/my/venv/bin/activate

Instructions
============

To install the package, you can use the following command:

.. code:: bash

   python -m pip install anemoi-datasets

If you are interested in creating datasets, you can install the package
with the following command:

.. code:: bash

   pip install anemoi-datasets[create]

For an editable install of anemoi-datasets, you can use the following
command. In this case, changes that you make to the anemoi-datasets code
will be reflected in the installed package without having to reinstall
it.

.. code:: bash

   pip install -e .

We also maintain other dependency sets for different subsets of
functionality:

.. code:: bash

   python -m pip install "anemoi-datasets[docs]" # Install optional dependencies for generating docs

.. literalinclude:: ../pyproject.toml
   :language: toml
   :start-at: optional-dependencies.all
   :end-before: urls.Changelog

**********************
 Development versions
**********************

To install the most recent development version, install from GitHub:

.. code::

   $ python -m pip install git@github.com:ecmwf/anemoi-datasets.git

*********
 Testing
*********

To run the test suite after installing anemoi-datasets, install (via
PyPI) `py.test <https://pytest.org>`__ and run ``pytest`` in the
``datasets`` directory of the anemoi-datasets repository.
