####################################
 Welcome to Anemoi's documentation!
####################################

.. warning::

   This documentation is work in progress. It is not yet ready.
   Currently, the documentation is based on the one from the ecml-tools_
   project, which will be merged into Anemoi.

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

An *Anemoi dataset* is a thin wrapper around a zarr_ store that is
optimised for training data-driven weather forecasting models. It is
organised in such a way that I/O operations are minimised.

This documentation is divided into two main sections: :ref:`how to use
existing datasets <using-introduction>` and :ref:`how to build new
datasets <building-introduction>`.

-  :doc:`overview`
-  :doc:`installing`
-  :doc:`firststeps`
-  :doc:`examples`

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installing
   firststeps
   examples

**Using training datasets**

-  :doc:`using/introduction`
-  :doc:`using/opening`
-  :doc:`using/subsetting`
-  :doc:`using/combining`
-  :doc:`using/selecting`
-  :doc:`using/grids`
-  :doc:`using/statistics`
-  :doc:`using/other`
-  :doc:`using/options`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Using datasets

   using/introduction
   using/opening
   using/subsetting
   using/combining
   using/selecting
   using/grids
   using/statistics
   using/other
   using/options

**Building training datasets**

-  :doc:`building/introduction`
-  :doc:`building/operations`
-  :doc:`building/sources`
-  :doc:`building/filters`
-  :doc:`building/statistics`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Building datasets

   building/introduction
   building/operations
   building/sources
   building/filters
   building/naming_variables
   building/handling_missing_dates
   building/handling_missing_values
   building/statistics

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

.. _ecml-tools: https://github.com/ecmwf-lab/ecml-tools

.. _zarr: https://zarr.readthedocs.io/
