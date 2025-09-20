########
 rename
########

When combining several sources, it is common to have different values
for a given attribute to represent the same concept. For example,
``temperature_850hPa`` and ``t_850`` are two different ways to represent
the temperature at 850 hPa. The ``rename`` filter allows renaming a key
to another key. It is a :ref:`filter <filters>` that must follow a
:ref:`source <sources>` or another filter in a :ref:`building-pipe`
operation.

.. literalinclude:: yaml/rename.yaml
   :language: yaml

.. note::

   The ``rename`` filter was primarily designed to rename the ``param``
   attribute, but any key can be renamed. The ``rename`` filter can take
   several renaming keys.
