.. _miscellaneous:

#########################
 Miscellaneous functions
#########################

The two functions below can be used modify temporarily the
:ref:`configuration <configuration>` so that the packages can find named
datasets at given locations.

Use ``add_dataset_path`` to add a path to the list of paths where the
package searches for datasets:

.. _add_dataset_path:

.. literalinclude:: code/misc1.py
   :language: python

Use ``add_named_dataset`` to add a named dataset to the list of named
datasets:

.. _add_named_dataset:

.. literalinclude:: code/misc2.py
   :language: python
