################
 repeated-dates
################

The `repeated-dates` source is used to repeat a single source multiple
times, so that its data are present on multiple dates. A simple example
of this is when you have a source that contains a constant field, such
as orography or bathymetry, that you want to have repeated on all the
dates of the dataset.

The general format of the `repeated-dates` source is:

.. literalinclude:: yaml/repeated_dates1.yaml
   :language: yaml

where ``source`` is any of the :ref:`operations <operations>` or
:ref:`sources <sources>` described in the previous sections. The
``mode`` parameter can be one of the following:

**********
 constant
**********

.. literalinclude:: yaml/repeated-dates2.yaml
   :language: yaml

*************
 climatology
*************

.. literalinclude:: yaml/repeated-dates3.yaml
   :language: yaml

*********
 closest
*********

.. literalinclude:: yaml/repeated-dates4.yaml
   :language: yaml
