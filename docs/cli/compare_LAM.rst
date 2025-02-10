compareLAM
==========

Compare statistic of two datasets. 
This command compares the statistics of each variable in two datasets ONLY in the overlapping area between the two.

Example of useful applications: 
 - Stretched Grid
 - Boundary LAM

In both application, it's necessary to check allignemnt between the variables of the local dataset and the ones of the global.
Datasets will cohexist on the same grid and statistical coherence is essential for training stability.

The `compareLAM` will output a table showcasing the comparison of dataset statistics in html format. 
Additionally a plot of the datasets grids can be displayed.


USAGE:
******
.. code:: console

   $ anemoi-datasets compareLAM dataset1 dataset2 num_dates outpath round_ndigits --save_plots
where:
 - dataset1: path to the first dataset (the global dataset)
 - dataset2: path to the second dataset (the LAM dataset)
 - num_dates: number of datapoints (in time) to compare over.
 - outpath: path to store the table (and the optional plots).
 - round_ndigits: number of decimal digits to keep in the table.
 - (optional) save_plots: activate to store an image of the dataset grids.


.. argparse::
    :module: anemoi.datasets.__main__
    :func: create_parser
    :prog: anemoi-datasets
    :path: compareLAM