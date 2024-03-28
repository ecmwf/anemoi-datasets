.. _combining-datasets:

####################
 Combining datasets
####################

When combining datasets, the statistics of the first dataset are used by
default. You can change this by setting the :ref:`selecting-statistics`
option to a different dataset, even if it is not part of the
combination.

When combining datasets, the package will check that the datasets are
compatible, i.e. that they have the same resolution, the same variables,
etc. The compatibility checks depend on the type of combination. You can
adjust some of the attributes of the datasets to make them compatible,
e.g. by changing their date range or frequency using :ref:`start`,
:ref:`end`, :ref:`frequency`, etc. You can also ask the package to
:ref:`automatically adjust <using-matching>` these attributes.

.. _concat:

********
 concat
********

You can concatenate two or more datasets along the dates dimension. The
package will check that all datasets are compatible (same resolution,
same variables, etc.). Currently, the datasets must be given in
chronological order with no gaps between them.

.. literalinclude:: code/concat1.py
   :language: python

.. image:: images/concat.png
   :align: center
   :alt: Concatenation

Please note that you can pass more than two ``zarr`` files to the
function.

   **NOTE:** When concatenating file, the statistics are not recomputed;
   it is the statistics of first file that are returned to the user.

.. _join:

******
 join
******

You can join two datasets that have the same dates, combining their
variables.

.. literalinclude:: code/join1.py
   :language: python

.. image:: images/join.png
   :align: center
   :alt: Join

If a variable is present in more that one file, that last occurrence of
that variable will be used, and will be at the position of the first
occurrence of that name.

.. image:: images/overlay.png
   :align: center
   :alt: Overlay

Please note that you can join more than two ``zarr`` files.

.. _ensembles:

***********
 ensembles
***********

You can combine two or more datasets that have the same dates,
variables, grids, etc. along the ensemble dimension. The package will
check that all datasets are compatible.

.. literalinclude:: code/ensembles1_.py

.. _grids:

*******
 grids
*******

.. literalinclude:: code/grids1_.py

The values for ``mode`` are:

mode=concatenate
================

All the grid points are concatenated, in the order they are given. The
`latitudes` and `longitudes` are also concatenated.

mode=cutout
===========

The `cutout` mode only supports two datasets. The first dataset is the
considered to be a limited area model (LAM), while the second one is
considered to be a global model or boundary conditions. It is therefore
expected that the bounding box of the first dataset is contained within
the bounding box of the second dataset.

The image below shows the global dataset:

.. image:: images/cutout-1.png
   :width: 75%
   :align: center
   :alt: Cutout

The image below shows the LAM dataset:

.. image:: images/cutout-2.png
   :width: 75%
   :align: center
   :alt: Cutout

A 'cutout' is performed by removing the grid points from the global
dataset that contained in the LAM dataset. The result is shown below:

.. image:: images/cutout-3.png
   :width: 75%
   :align: center
   :alt: Cutout

The final dataset is the concatenation of the LAM dataset and the
cutout:

.. image:: images/cutout-4.png
   :width: 75%
   :align: center
   :alt: Cutout
