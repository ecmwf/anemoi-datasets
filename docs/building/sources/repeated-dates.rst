################
 repeated-dates
################

The `repeated-dates` source is used to repeat a single source multiple
times, so that its data are present on multiple dates. A simple example
of this is when you have a source that contains a constant field, such
as orography or bathymetry, that you want to have repeated on all the
dates of the dataset.

The general format of the `repeated-dates` source is:

.. literalinclude:: yaml/repeated-dates1.yaml
   :language: yaml

where ``source`` is any of the :ref:`operations <operations>` or
:ref:`sources <sources>` described in the previous sections. The
``mode`` parameter can be one of the following:

**********
 constant
**********

.. literalinclude:: yaml/repeated-dates2.yaml
   :language: yaml

The ``constant`` mode accepts an optional ``date:`` key:

- If ``date:`` is **set** to a specific datetime, the inner source is
  invoked for that single date and the resulting fields are broadcast
  over all of the recipe's dates. The value is validated at recipe-parse
  time so an invalid string fails fast. Example:

  .. code:: yaml

     repeat-dates:
       mode: constant
       date: 2020-01-01 00:00:00
       source:
         mars:
           class: od
           param: z
           type: an
           levtype: sfc

- If ``date:`` is **null** (or omitted), the inner source is invoked
  with an empty date list — which is the supported way to plug in a
  fixed, fully-specified MARS request whose ``date``/``time`` are
  already baked into the source block. This is the typical pattern for
  dateless forcings such as a single orography field retrieved with its
  own hard-coded MARS coordinates.

  .. code:: yaml

     repeat-dates:
       mode: constant
       date: null
       source:
         mars:
           class: od
           expver: "0001"
           stream: oper
           type: an
           levtype: sfc
           param: z
           date: 20200101
           time: 0

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
