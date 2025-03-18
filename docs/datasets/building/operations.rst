.. _operations:

############
 Operations
############

Operations are blocks of YAML code that translates a list of dates into
fields.

.. _building-join:

******
 join
******

The join is the process of combining several sources data. Each source
is expected to provide different variables at the same dates.

.. literalinclude:: yaml/input.yaml
   :language: yaml

.. _building-concat:

********
 concat
********

The concatenation is the process of combining different sets of
operation that handle different dates. This is typically used to build a
dataset that spans several years, when the several sources are involved,
each providing a different period.

.. literalinclude:: yaml/concat.yaml
   :language: yaml

.. _building-pipe:

******
 pipe
******

The pipe is the process of transforming fields using :ref:`filters
<filters>`. The first step of a pipe is typically a source, a join or
another pipe. The following steps are filters.

.. literalinclude:: yaml/pipe.yaml
   :language: yaml
