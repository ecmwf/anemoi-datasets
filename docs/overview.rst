.. _overview:

##########
 Overview
##########

Training datasets are large array-like objects encode in Zarr_ format.
They

The array has the following dimensions:

.. figure:: overview.png
   :alt: Data layout
   :align: center

The first dimension is the time dimension, the second dimension are the
variables (e.g. temperature, pressure, etc), the third dimension is the
ensemble, and fourth dimension are the grid points values.

.. figure:: matrix.svg
   :alt: Data chunking
   :align: center

This structure provides an efficient way to build the training dataset,
as input and output of the model are simply consecutive slices of the
array.

.. literalinclude:: overview_.py
   :language: python




.. _zarr: https://zarr.readthedocs.io/
