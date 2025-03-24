################
 repeated_dates
################

The `repeated_dates` source is used to repeat a single source multiple
times, so that its data is present at multiple dates. A simple example
of this is when you have source that contains a constant field, such as
orography or bathymetry, that you want to have repeated at all the dates
of the dataset.

The generale format of the `repeated_dates` source is:

.. literalinclude:: yaml/repeated_dates1.yaml
   :language: yaml

where ``source`` is any of the :ref:`operations <operations>` or
:ref:`sources <sources>` described in the previous sections. The
``mode`` parameter can be one of the following:

**********
 constant
**********

.. literalinclude:: yaml/repeated_dates2.yaml
   :language: yaml

*************
 climatology
*************

.. literalinclude:: yaml/repeated_dates3.yaml
   :language: yaml

*********
 closest
*********

.. literalinclude:: yaml/repeated_dates4.yaml
   :language: yaml
