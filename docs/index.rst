.. _index-page:

#############################################
 Welcome to `anemoi-datasets` documentation!
#############################################

.. warning::

   This documentation is work in progress.

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

.. toctree::
   :maxdepth: 1
   :hidden:

   overview
   installing

**Using training datasets**

-  :doc:`using/introduction`
-  :doc:`using/opening`
-  :doc:`using/methods`
-  :doc:`using/subsetting`
-  :doc:`using/combining`
-  :doc:`using/selecting`
-  :doc:`using/grids`
-  :doc:`using/zip`
-  :doc:`using/statistics`
-  :doc:`using/missing`
-  :doc:`using/other`
-  :doc:`using/matching`
-  :doc:`using/miscellaneous`
-  :doc:`using/configuration`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Using datasets

   using/introduction
   using/opening
   using/methods
   using/subsetting
   using/combining
   using/selecting
   using/grids
   using/zip
   using/statistics
   using/missing
   using/other
   using/matching
   using/miscellaneous
   using/configuration

**Building training datasets**

-  :doc:`building/introduction`
-  :doc:`building/operations`
-  :doc:`building/sources`
-  :doc:`building/filters`
-  :doc:`building/statistics`
-  :doc:`building/incremental`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Building datasets

   building/introduction
   building/operations
   building/sources
   building/filters
   building/naming-variables
   building/handling-missing-dates
   building/handling-missing-values
   building/statistics
   building/incremental
   building/advanced-options

**Command line tool**

-  :doc:`cli/introduction`
-  :doc:`cli/create`
-  :doc:`cli/inspect`
-  :doc:`cli/compare`
-  :doc:`cli/copy`

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Command line tool

   cli/introduction
   cli/create
   cli/inspect
   cli/compare
   cli/copy

*****************
 Anemoi packages
*****************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-graphs <anemoi-graphs:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

.. _ecml-tools: https://github.com/ecmwf-lab/ecml-tools

.. _zarr: https://zarr.readthedocs.io/
