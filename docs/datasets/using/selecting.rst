.. _selecting-variables:

#####################
 Selecting variables
#####################

Selecting is the action of filtering the dataset by its second dimension
(variables).

.. _select:

********
 select
********

Select '2t' and 'tp' in that order:

.. code:: python

   ds = open_dataset(dataset, select=["2t", "tp"])

Select '2t' and 'tp', but preserve the order in which they are in the
dataset:

.. code:: python

   ds = open_dataset(dataset, select={"2t", "tp"})

.. _drop:

******
 drop
******

You can also drop some variables:

.. code:: python

   ds = open_dataset(dataset, drop=["10u", "10v"])

.. _reorder:

*********
 reorder
*********

and reorder them:

... using a list:

.. code:: python

   ds = open_dataset(
       dataset,
       reorder=["2t", "msl", "sp", "10u", "10v"],
   )

... or using a dictionary:

.. code:: python

   ds = open_dataset(
       dataset,
       reorder={
           "2t": 0,
           "msl": 1,
           "sp": 2,
           "10u": 3,
           "10v": 4,
       },
   )

.. _rename:

********
 rename
********

You can also rename variables:

.. code:: python

   ds = open_dataset(dataset, rename={"2t": "t2m"})

This will be useful when you join datasets and do not want variables
from one dataset to override the ones from the other.

********
 number
********

If a dataset is an ensemble, you can select one or more specific members
using the `number` option. You can also use ``numbers`` (which is an
alias for ``number``), and ``member`` (or ``members``). The difference
between the two is that ``number`` is **1-based**, while ``member`` is
**0-based**.

Select a single element:

.. code:: python

   ds = open_dataset(
       dataset,
       number=1,
   )

... or a list:

.. code:: python

   ds = open_dataset(
       dataset,
       number=[1, 3, 5],
   )

.. _rescale:

*********
 rescale
*********

When combining datasets, you may want to rescale the variables so that
they have matching units. This can be done with the `rescale` option:

.. code:: python

   # Scale and offset can be passed as a dictionary...

   ds = open_dataset(
       dataset,
       rescale={"2t": {"scale": 1.0, "offset": -273.15}},
   )

   # ... a tuple of floating points ...

   ds = open_dataset(
       dataset,
       rescale={"2t": (1.0, -273.15)},
   )

   # ... or a tuple of strings representing units.

   ds = open_dataset(
       dataset,
       rescale={"2t": ("K", "degC")},
   )

   # Several variables can be rescaled at once.

   ds = open_dataset(
       dataset,
       rescale={
           "2t": ("K", "degC"),
           "tp": ("m", "mm"),
       },
   )

The `rescale` option will also rescale the statistics. The rescaling is
currently limited to simple linear conversions.

When provided with units, the `rescale` option uses the cfunits_ package
to find the `scale` and `offset` attributes of the units and uses these
to rescale the data.

.. warning::

   When providing units, the library assumes that the mapping between
   them is a linear transformation. No check is done to ensure this is
   the case.

.. _cfunits: https://github.com/NCAS-CMS/cfunits

.. _number:
