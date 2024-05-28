.. _opening-datasets:

##################
 Opening datasets
##################

The simplest way to open a dataset is to use the `open_dataset`
function:

.. literalinclude:: code/open_first_.py
   :language: python

In that example, `dataset` can be:

-  a local path to a dataset on disk:

.. literalinclude:: code/open_path.py
   :language: python

-  a URL to a dataset in the cloud:

.. literalinclude:: code/open_cloud.py
   :language: python

-  a dataset name, which is a string that identifies a dataset in the
   `anemoi` :ref:`configuration file <configuration>`.

.. literalinclude:: code/open_name.py
   :language: python

-  an already opened dataset. In that case, the function uses the
   options to return a modified dataset, for example with a different
   time range or frequency.

.. literalinclude:: code/open_other.py
   :language: python

-  a dictionary with a ``dataset`` key that can be any of the above, and
   the remaining keys being the options. The purpose of this option is
   to allow the user to open a dataset based on a configuration file.
   See :ref:`an example <open_with_config>` below:

.. literalinclude:: code/open_dict_.py
   :language: python

-  a list of any of the above that will be combined either by
   concatenation or joining, based on their compatibility.

.. literalinclude:: code/open_list_.py
   :language: python

-  a combining keyword, such as `join`, `concat`, `ensembles`, etc.
   followed by a list of the above. See :ref:`combining-datasets` for
   more information.

.. literalinclude:: code/open_combine1_.py
   :language: python

.. note::

   In the example above, the options `option1`, `option2`, apply to the
   combined dataset. To apply options to individual datasets, use a list
   of dictionaries as shown below. The options `option1`, `option2`,
   apply to the first dataset, and `option3`, `option4`, to the second
   dataset, etc.

.. literalinclude:: code/open_combine2_.py
   :language: python

.. _open_with_config:

As mentioned above, using the dictionary to open a dataset can be useful
for software that provides users with the ability to define their
requirements in a configuration file:

.. literalinclude:: code/open_yaml_.py
   :language: python

The dictionary can be as complex as needed, for example:

.. literalinclude:: code/open_complex.py
   :language: python

..
   TODO:
   When opening a complex dataset the user can use the `adjust` keyword to
   let the function know how to combine the datasets. The `combine` keyword
   can be any of the following:
