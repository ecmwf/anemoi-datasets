.. _opening-datasets:

##################
 Opening datasets
##################

The simplest way to open a dataset is to use the `open_dataset`
function:

.. literalinclude:: code/opening_basic_.py

In this example, `dataset` can be:

-  a local path to a dataset on disk:

.. literalinclude:: code/opening_path.py

-  a URL to a dataset in the cloud:

.. literalinclude:: code/opening_url.py

-  a dataset name, which is a string that identifies a dataset in the
   `anemoi` :ref:`configuration file <configuration>`.

.. literalinclude:: code/opening_name.py

-  an already opened dataset. In that case, the function uses the
   options to return a modified dataset, for example with a different
   time range or frequency.

.. literalinclude:: code/opening_reopen.py

-  a dictionary with a ``dataset`` key that can be any of the above, and
   the remaining keys being the options. The purpose of this option is
   to allow the user to open a dataset based on a configuration file.
   See :ref:`an example <open_with_config>` below:

.. literalinclude:: code/opening_dict_.py

-  a list of any of the above that will be combined either by
   concatenation or joining, based on their compatibility.

.. literalinclude:: code/opening_list_.py

-  a combining keyword, such as `join`, `concat`, `ensembles`, etc.
   followed by a list of the above. See :ref:`combining-datasets` for
   more information.

.. literalinclude:: code/opening_combine_keyword_.py

.. note::

   In the example above, the options `option1`, `option2`, apply to the
   combined dataset. To apply options to individual datasets, use a list
   of dictionaries as shown below. The options `option1`, `option2`,
   apply to the first dataset, and `option3`, `option4`, to the second
   dataset, etc.

.. literalinclude:: code/opening_combine_list_.py

.. _open_with_config:

As mentioned above, using the dictionary to open a dataset can be useful
for software that provides users with the ability to define their
requirements in a configuration file:

.. literalinclude:: code/opening_config.py

The dictionary can be as complex as needed, for example:

.. literalinclude:: code/opening_complex_config.py

The `open_dataset` function returns an object that wraps around
`numpy.ndarray`, so it is possible to inspect the dataset and visualise
it with standard Python tools. For example:

.. literalinclude:: code/opening_plot.py

.. figure:: ../_static/2t_map_example.png
   :alt: example map plot
   :align: center

..
   TODO:
   When opening a complex dataset the user can use the `adjust` keyword to
   let the function know how to combine the datasets. The `combine` keyword
   can be any of the following:
