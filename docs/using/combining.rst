.. _combining-datasets:

####################
 Combining datasets
####################

When combining datasets, the statistics of the first dataset are used by
default. You can change this by setting the :ref:`selecting-statistics`
option to a different dataset, even if it is not part of the
combination. See

.. _concat:

********
 concat
********

You can concatenate two or more datasets along the dates dimension. The
package will check that all datasets are compatible (same resolution,
same variables, etc.). Currently, the datasets must be given in
chronological order with no gaps between them.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1940-1978-1h-v2",
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
   )

.. image:: concat.png
   :alt: Concatenation

Please note that you can pass more than two ``zarr`` files to the
function.

   **NOTE:** When concatenating file, the statistics are not recomputed;
   it is the statistics of first file that are returned to the user.

******
 join
******

You can join two datasets that have the same dates, combining their
variables.

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(
       "aifs-ea-an-oper-0001-mars-o96-1979-2022-1h-v2",
       "some-extra-parameters-from-another-source-o96-1979-2022-1h-v2",
   )

.. image:: join.png
   :alt: Join

If a variable is present in more that one file, that last occurrence of
that variable will be used, and will be at the position of the first
occurrence of that name.

.. image:: overlay.png
   :alt: Overlay

Please note that you can join more than two ``zarr`` files.

***********
 ensembles
***********

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(ensembles=[dataset1, dataset2, ...])

*******
 grids
*******

.. code:: python

   from anemoi.datasets import open_dataset

   ds = open_dataset(grids=[dataset1, dataset2, ...], method=...)
