.. _index-page:

#############################################
 Welcome to `anemoi-datasets` documentation!
#############################################

.. warning::

   This documentation is work in progress.

An *Anemoi dataset* is a thin wrapper around a zarr_ store that is
optimised for training data-driven weather forecasting models.
anemoi-datasets are organised in such a way that I/O operations are
minimised. It is one of the packages within the `anemoi framework
<https://anemoi-docs.readthedocs.io/en/latest/>`_.

**************
 About Anemoi
**************

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

****************
 Quick overview
****************

#!TODO

This package provides the *Anemoi* datasets functionality.

.. toctree::
   :maxdepth: 3
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
   :maxdepth: 3
   :caption: User Guide

   datasets/introduction
   datasets/building/introduction
   datasets/using/introduction

.. toctree::
   :maxdepth: 5
   :hidden:
   :caption: CLI

   cli/create
   cli/inspect
   cli/compare
   cli/copy
   cli/scan
   cli/patch
   cli/requests

.. toctree::
   :maxdepth: 1
   :glob:
   :caption: Api Reference

   modules/*

.. toctree::
   :maxdepth: 3
   :caption: Developing Anemoi Datasets

   dev/contributing
   dev/code_structure
   dev/testing

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

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

.. _zarr: https://zarr.readthedocs.io/
