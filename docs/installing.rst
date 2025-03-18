############
 Installing
############

****************
 Using datasets
****************

To install the latest stable release of the package with its
dependencies, you can use the following command:

.. code:: bash

   pip install anemoi-datasets

*******************
 Creating datasets
*******************

.. code:: bash

   pip install anemoi-datasets[create]

**************
 Contributing
**************

.. code:: bash

   git clone git@github.com:ecmwf/anemoi-datasets.git
   cd anemoi-datasets
   pip install .[dev]
   pip install -r docs/requirements.txt

You may also have to install pandoc on MacOS:

.. code:: bash

   brew install pandoc

For an editable install of anemoi-datasets, you can use the following
command. In this case, changes that you make to the anemoi-datasets code
will be reflected in the installed package without having to reinstall
it.

.. code:: bash

   pip install -e .

..
   TODO: Make sure to update `setup.py`
   to reflect these options
