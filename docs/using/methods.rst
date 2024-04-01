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

dtype:
   The dataset's data type.

field_shape:
   The dataset's field shape.

dates:
   The dataset's dates.

latitudes:
   The dataset's latitudes.

longitudes:
   The dataset's longitudes.

statistics:
   The dataset's statistics.

resolution:
   The dataset's resolution.

frequency:
   The dataset's frequency.

name_to_index:
   A dictionary mapping variable names to their indices.

variables:
   A list of the dataset's variable names.

missing:
   The index of the missing dates.

grids:
   The dataset's grids.
