.. _using-selecting:

#####################
 Selecting variables
#####################

Selecting is the action of filtering the dataset by its second dimension
(variables).

.. _select:

********
 select
********

If you pass a **list**, the resulting dataset contains the requested
variables in the order given by the list. Here, ``2t`` comes first and
``tp`` second, regardless of their order in the original dataset:

.. literalinclude:: code/selecting_select_list_.py

If you pass a **set**, the resulting dataset contains the same variables,
but their original order in the dataset is preserved (sets are unordered,
so they cannot impose an order):

.. literalinclude:: code/selecting_select_set_.py

.. _drop:

******
 drop
******

You can also drop some variables:

.. literalinclude:: code/selecting_drop_.py

.. _reorder:

*********
 reorder
*********

and reorder them:

... using a list:

.. literalinclude:: code/selecting_reorder_list_.py

... or using a dictionary:

.. literalinclude:: code/selecting_reorder_dict_.py

.. _rename:

********
 rename
********

You can also rename variables:

.. literalinclude:: code/selecting_rename_.py

This will be useful when you join datasets and do not want variables
from one dataset to override the ones from the other.

********
 number
********

If a dataset is an ensemble, you can select one or more specific members
using the `number` option. See :ref:`number` in the
:ref:`using-ensembles` section for details.

.. _rescale:

*********
 rescale
*********

When combining datasets, you may want to rescale the variables so that
they have matching units. This can be done with the `rescale` option:

.. literalinclude:: code/selecting_rescale_.py

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
