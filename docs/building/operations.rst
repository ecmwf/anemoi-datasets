.. _operations:

############
 Operations
############

Operations are blocks of YAML code that translates a list of dates into
fields.

******
 join
******

The join is the process of combining several sources data. Each source
is expected to provide different variables at the same dates.

.. literalinclude:: input.yaml
   :language: yaml

********
 concat
********

The concatenation is the process of combining different sets of
operation that handle different dates. This is typically used to build a
dataset that spans several years, when the several sources are involved,
each providing a different period.

.. literalinclude:: concat.yaml
   :language: yaml

******
 pipe
******

The pipe is the process of transforming fields using :ref:`filters
<filters>`. The first step of a pipe is typically a source, a join or
another pipe. The following steps are filters.

.. literalinclude:: pipe.yaml
   :language: yaml
