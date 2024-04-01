########################
 Methods and attributes
########################

The following methods and attributes are available for the objects
returned by ``open_dataset``.

.. warning::

   All methods and attributes will take into account any subsetting,
   selecting or combining used to construct the final dataset with the
   exception of ``statistics`` which will return the values of the first
   dataset encountered. See :ref:`statistics` for more details.

*********
 Methods
*********

__len__()
   The number of rows (dates) in the dataset.

__getitem__(key)
   Access the dataset's data values.

metadata()
   Return the dataset's metadata.

provenance()
   Return the dataset's provenance information.

source(index):
   For debugging. Given the index of variable, this will return from
   which Zarr store it will be loaded. This is useful to debug combining
   datasets with :ref:`join`.

tree():
   For debugging. Return the dataset's internal tree structure.

************
 Attributes
************

shape:
   A tuple of the dataset's dimensions.

field_shape:
   The original shape of a single field, either 1D or 2D. When building
   datasets, the fields are flattened to 1D.

dtype:
   The dataset's `NumPy data type`_.

dates:
   The dataset's dates, as a NumPy vector of datetime64_ objects.

frequency:
   The dataset's frequency (i.e the delta between two consecutive dates)
   in hours.

latitudes:
   The dataset's latitudes as a NumPy vector.

longitudes:
   The dataset's longitudes as a NumPy vector.

statistics:
   The dataset's statistics.

resolution:
   The dataset's resolution.

name_to_index:
   A dictionary mapping variable names to their indices.

   .. code:: python

      print(dataset.name_to_index["2t"])

variables:
   A list of the dataset's variable names, in the order they appear in the
      dataset.

missing:
   The set of indices of the missing dates.

grids:
   A tuple of number of grid point for each datasets that are combined
   with the :ref:`grid` method.

.. _datetime64: https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html

.. _numpy data type: https://docs.scipy.org/doc/numpy/user/basics.types.html
