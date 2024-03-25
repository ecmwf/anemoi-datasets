.. _selecting-other:

##################
 Other operations
##################

.. warning::

   The operations described in this section are do not check that their
   inputs are compatible.

*****
 zip
*****

The `zip` operation is used to combine multiple datasets into a single
dataset.

.. code:: python

   ds = open_dataset(zip=[dataset1, dataset2, ...])

This operation is similar to the Python's :py:func:`zip` function, but
it returns tuples of the selected indices instead of the values:

.. code:: python

   print(ds[0])
   # (dataset1[0], dataset2[0], ...)

   print(ds[0, 1])
   # (dataset1[0, 1], dataset2[0, 1], ...)

   print(ds[0:2])
   # (dataset1[0:2], dataset2[0:2], ...)

*******
 chain
*******

.. code:: python

   ds = open_dataset(chain=[dataset1, dataset2, ...])

The `chain` operation is used to combine multiple datasets into a single
dataset. The datasets are combined by concatenating the data arrays
along the first dimension (dates). This is similar to the :ref:`concat`
operation, but no check are done to see if the datasets are compatible,
this means that the shape of the arrays returned when iterating or
indexing may be different.

This operation is identical to the Python's :py:func:`itertools.chain`
function.

*********
 shuffle
*********

.. code:: python

   ds = open_dataset(dataset, shuffle=True)

The `shuffle` operation is used to shuffle the data in the dataset along
the first dimension (dates).
