.. _operations:

############
 Operations
############

Operations are blocks of YAML code that translate a list of dates into
fields.

.. _building-join:

******
 join
******

The join is the process of combining data from several sources. Each
source is expected to provide different variables for the same dates.

.. literalinclude:: ../yaml/input.yaml
   :language: yaml

.. _building-concat:

********
 concat
********

Concatenation is the process of combining different sets of operations
that handle different dates. This is typically used to build a dataset
that spans several years, when several sources are involved, each
providing a different period.

.. literalinclude:: ../yaml/concat.yaml
   :language: yaml

.. _building-pipe:

******
 pipe
******

The pipe is the process of transforming fields using :ref:`filters
<filters>`. The first step of a pipe is typically a source, a join, or
another pipe. The following steps are filters.

.. literalinclude:: ../yaml/pipe.yaml
   :language: yaml
