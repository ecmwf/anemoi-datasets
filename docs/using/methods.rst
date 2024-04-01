########################
 Methods and attributes
########################

.. warning::

   Page in progress.

The following methods and attributes are available for the objects
returned by ``open_dataset``:

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
   Return the source of the dataset's data.

tree():
   Return the dataset's tree.

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

variables:
   A list of the dataset's variable names.

missing:
   The index of the missing dates.

grids:
   The dataset's grids.

.. _datetime64: https://docs.scipy.org/doc/numpy/reference/arrays.datetime.html

.. _numpy data type: https://docs.scipy.org/doc/numpy/user/basics.types.html
