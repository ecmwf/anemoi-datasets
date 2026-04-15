.. _selecting-members:

###################
 Selecting members
###################

This section describes how to subset data that are part of an ensemble.
To combine ensembles, see :ref:`ensembles` in the
:ref:`combining-datasets` section.

.. _number:

If a dataset is an ensemble, you can select one or more specific members
using the `number` option. You can also use ``numbers`` (which is an
alias for ``number``), and ``member`` (or ``members``). The difference
between the two is that ``number`` is **1-based**, whereas ``member`` is
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
