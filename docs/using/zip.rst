.. _selecting-zip:

#################################
 Iterating over several datasets
#################################

Sometimes you need to iterate over several datasets at the same time.
The following functions will help you with that.

.. warning::

   When iterating over several datasets, most of the functions and
   properties of the dataset object returned by :py:func:`open_dataset`
   will not return a :py:class:`tuple` instead of a single value. The
   tuple will contain the values of the corresponding datasets. There
   are a few exceptions, such as `dates` or `missing`.

**************
 x=..., y=...
**************

In machine learning, you often need to iterate over two datasets at the
same time. One representing the input ``x`` and the other the output
``y``.

.. literalinclude:: code/xy1_.py

You will then be able to iterate over the datasets as follows:

.. literalinclude:: code/xy2_.py

**Note:** `xy` is currently a shortcut for `zip` below, and is intended
to make the code more readable.

*****
 zip
*****

The `zip` option is used to combine multiple datasets into a single
dataset, that can be iterated over or indexed simultaneously.

.. literalinclude:: code/zip1_.py

The dataset can then be indexed as follows:

.. literalinclude:: code/zip2_.py

.. note::

   The `zip` option is similar to Python's :py:func:`zip` function. The
   main difference the datasets are checked to be compatible before
   being combined (same ranges of dates, same frequency, etc.). Also,
   Python's :py:func:`zip` only allows iteration, while the `zip` option
   allows indexing as well.

****************
 Combining both
****************

Both options can be combined. The example below is based on a model that
is trained to upscale a dataset. The input is the low resolution and a
high resolution orography. The output is the high resolution dataset.

.. literalinclude:: code/zip_xy_.py
