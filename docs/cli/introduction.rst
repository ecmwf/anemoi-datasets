.. _cli-introduction:

############
Introduction
############

When you install the `anemoi-datasets` package, this will also install command line tool
called ``anemoi-datasets`` which can be used to manage the zarr datasets.

The tool can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-datasets --help

The commands are:

- :ref:`Create Command <create_command>`
- :ref:`Copy Command <copy_command>`
- :ref:`Inspect Command <Inspect_command>`
- :ref:`Compare Command <compare_command>`
- :ref:`Scan Command <scan_command>`
- :ref:`Validate Command <validate_command>`
- :ref:`Compare LAM Command <compare_lam_command>`




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
   cli/validate
