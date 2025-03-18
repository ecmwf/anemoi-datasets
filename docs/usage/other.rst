.. _selecting-other:

##################
 Other operations
##################

.. warning::

   The operations described in this section do not check that their
   inputs are compatible.

*******
 chain
*******

.. literalinclude:: code/chain_.py

The `chain` operation is used to combine multiple datasets into a single
dataset. The datasets are combined by concatenating the data arrays
along the first dimension (dates). This is similar to the :ref:`concat`
operation, but no checks are done to see if the datasets are compatible,
this means that the shape of the arrays returned when iterating or
indexing may be different.

This operation is identical to Python's :py:func:`itertools.chain`
function.

*********
 shuffle
*********

.. literalinclude:: code/shuffle_.py

The `shuffle` operation is used to shuffle the data in the dataset along
the first dimension (dates).
