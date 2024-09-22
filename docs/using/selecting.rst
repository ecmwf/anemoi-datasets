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

.. literalinclude:: code/select1_.py
   :language: python

Select '2t' and 'tp', but preserve the order in which they are in the
dataset

.. literalinclude:: code/select2_.py
   :language: python

.. _drop:

******
 drop
******

You can also drop some variables:

.. literalinclude:: code/drop_.py
   :language: python

.. _reorder:

*********
 reorder
*********

and reorder them:

... using a list

.. literalinclude:: code/reorder1_.py
   :language: python

... or using a dictionary

.. literalinclude:: code/reorder2_.py
   :language: python

.. _rename:

********
 rename
********

You can also rename variables:

.. literalinclude:: code/rename_.py
   :language: python

This will be useful when you join datasets and do not want variables
from one dataset to override the ones from the other.

*********
 rescale
*********

When combining datasets, you may want to rescale the variables so that
their have matching units. This can be done with the `rescale` option:

.. literalinclude:: code/rescale_.py
   :language: python

The `rescale` option will also rescale the statistics. The rescaling is
currently limited to simple linear conversions.

When provided with units, the `rescale` option uses the cfunits_ package
find the `scale` and `offset` attributes of the units and uses these to
rescale the data.

.. warning::

   When providing units, the library assumes that the mapping between
   them is a linear transformation. No check is does to ensure this is
   the case.

.. _cfunits: https://github.com/NCAS-CMS/cfunits
