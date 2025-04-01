.. _index-page:

#############################################
 Welcome to `anemoi-datasets` documentation!
#############################################

.. warning::

   This documentation is a work in progress.

An *Anemoi dataset* is a thin wrapper around a Zarr_ store that is
optimised for training data-driven weather forecasting models.
anemoi-datasets are organised in such a way that I/O operations are
minimised. It is one of the packages within the :ref:`anemoi framework
<anemoi-docs:index>`.

**************
 About Anemoi
**************

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises components or packages for preparing
training datasets, conducting ML model training, and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework, it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

****************
 Quick overview
****************

The anemoi-datasets package provides a structured approach to preparing
datasets for data-driven weather forecasting models, particularly those
using deep learning. By optimising data access patterns, anemoi-datasets
minimises I/O operations, improving efficiency when training machine
learning models.

anemoi-datasets offers a simple high-level interface based on a YAML
recipe file, which defines how datasets are processed and structured.
The package allows you to:

-  Load and transform datasets from sources such as reanalyses or
   forecasts.
-  Interpolate data to a desired spatial resolution and temporal
   frequency to match model requirements.
-  Select and preprocess relevant meteorological variables for use in
   machine learning workflows.
-  Structure datasets for efficient access in training and inference,
   reducing unnecessary data operations.

The dataset definition is specified in a YAML file, which is then used
to generate the dataset using the command-line tool :ref:`create command
<create_command>`. The command-line tool also allows users to inspect
datasets for compatibility with machine learning models.

In the rest of this documentation, you will learn how to configure and
create anemoi datasets using YAML files, as well as how to load and read
existing ones. A full example of a dataset preparation process can be
found in the :ref:`Create Your First Dataset <usage-getting-started>`
section.

************
 Installing
************

To install the package, you can use the following command:

.. code:: bash

   pip install anemoi-datasets

Get more information in the :ref:`installing <installing>` section.

**************
 Contributing
**************

.. code:: bash

   git clone ...
   cd anemoi-datasets
   pip install .[dev]

You may also have to install pandoc on macOS:

.. code:: bash

   brew install pandoc

***********************
 Other Anemoi packages
***********************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-transform <anemoi-transform:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-graphs <anemoi-graphs:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`
-  :ref:`anemoi-plugins <anemoi-plugins:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

.. _zarr: https://zarr.readthedocs.io/

..
   ..................................................................................

..
   From here defines the TOC in the sidebar, but is not rendered directly on the page.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction

   overview
   cli/introduction
   installing

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Recipe examples

   usage/getting_started

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: User Guide

   datasets/introduction
   datasets/building/introduction
   datasets/using/introduction

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: How-Tos

   howtos/introduction

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: CLI

   cli/create
   cli/inspect
   cli/grib-index
   cli/compare
   cli/copy
   cli/scan
   cli/patch
   cli/compare-lam

.. toctree::
   :maxdepth: 1
   :glob:
   :hidden:
   :caption: API Reference

   modules/*

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contributing

   dev/contributing
