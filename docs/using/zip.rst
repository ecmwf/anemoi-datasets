.. _selecting-zip:

#################################
 Iterating over several datasets
#################################

Sometimes you need to iterate over several datasets at the same time.
The following functions will help you with that.

.. warning::

   When iterating over several datasets, most of the functions and
   properties of the dataset object returned by :py:func:`open_dataset`
   will return a :py:class:`tuple` instead of a single value. The tuple
   will contain the values of the corresponding datasets. There are a
   few exceptions, such as `dates` or `missing`.

**************
 x=..., y=...
**************

In machine learning, you often need to iterate over two datasets at the
same time. One representing the input ``x`` and the other the output
``y``.

.. code:: python

   ds = open_dataset(x=dataset1, y=dataset2)

   # or

   ds = open_dataset(xy=[dataset1, dataset2])

You will then be able to iterate over the datasets as follows:

.. code:: python

   for x, y in ds:
       y_hat = model(x)
       loss = criterion(y_hat, y)
       loss.backward()

**Note:** `xy` is currently a shortcut for `zip` below, and is intended
to make the code more readable.

*****
 zip
*****

The `zip` option is used to combine multiple datasets into a single
dataset, that can be iterated over or indexed simultaneously.

.. code:: python

   ds = open_dataset(zip=[dataset1, dataset2, ...])

The dataset can then be indexed as follows:

.. code:: python

   print(ds[0])
   # (dataset1[0], dataset2[0], ...)

   print(ds[0, 1])
   # (dataset1[0, 1], dataset2[0, 1], ...)

   print(ds[0:2])
   # (dataset1[0:2], dataset2[0:2], ...)

.. note::

   The `zip` option is similar to Python's :py:func:`zip` function. The
   main difference is that the datasets are checked to be compatible
   before being combined (same ranges of dates, same frequency, etc.).
   Also, Python's :py:func:`zip` only allows iteration, while the `zip`
   option allows indexing as well.

****************
 Combining both
****************

Both options can be combined. The example below is based on a model that
is trained to upscale a dataset. The input is the low resolution and a
high resolution orography. The output is the high resolution dataset.

.. code:: python

   input = open_dataset(zip=[low_res_dataset, high_res_orography_dataset])
   output = open_dataset(high_res_dataset)

   ds = open_dataset(x=input, y=output)

   for (x, orography), y in ds:
       y_hat = model(x, orography)
       loss = criterion(y_hat, y)
       loss.backward()
