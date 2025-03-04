compare-lam
===========

Compare statistics of two datasets.
This command compares the statistics of each variable in two datasets **only in the overlapping area** between the two.

Example use cases:
------------------
- **Stretched Grid**
- **Boundary LAM**

In both cases, it is necessary to check the alignment between the variables of the local dataset and those of the global dataset.
Both datasets will coexist on the same grid, and statistical coherence is essential for training stability.

The `compare-lam` command outputs a table comparing dataset statistics in **HTML format**.
Additionally, a plot of the dataset grids can be displayed and saved if requested.

Usage:
******
.. code:: console

   $ anemoi-datasets compare-lam dataset1 dataset2 -nd num_dates -o outpath -rd round_ndigits --selected_vars var1 var2 ... [--save_plots]

Arguments:
----------

- **dataset1**: Path to the first dataset (the global dataset).
- **dataset2**: Path to the second dataset (the LAM dataset).
- **-nd, --num_dates**: Number of time steps (datapoints) to compare. *(default: 10)*
- **-o, --outpath**: Path to store the output table (and optional plots). *(default: "./")*
- **-rd, --round_ndigits**: Number of decimal places to round values to. *(default: 4)*
- **--selected_vars**: List of variables to compare between the datasets. *(default: ["10u", "10v", "2d", "2t"])*
- **--save_plots (optional)**: Enable this flag to save an image of the dataset grids.

Example:
--------

.. code:: console

   $ anemoi-datasets compare-lam aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v6.zarr \
     mch-co2-an-archive-0p02-2015-2020-6h-v3-pl13.zarr \
     -nd 10 -o "./" -rd 4 --selected_vars 2t msl --save_plots

Argparse integration:
---------------------

.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: compare-lam
