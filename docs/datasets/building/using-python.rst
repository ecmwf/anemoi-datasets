#############################
 Using Python define recipes
#############################

You can use Python to define recipes for building datasets. This allows
for more complex logic and flexibility compared to using static
configuration files.

When executed, the Python code will generate a YAML configuration that
can be used by the dataset building tool.

Here is an example of how to define a dataset recipe using Python.

First create a ``Recipe`` object, which will hold the configuration:

.. literalinclude:: code/using-python-1.py
   :language: python

you can pass parameters to the ``Recipe`` constructor:

.. literalinclude:: code/using-python-2.py
   :language: python

or set them later:

.. literalinclude:: code/using-python-3.py
   :language: python

You need to select which dates to use for building the dataset:

.. literalinclude:: code/using-python-4.py
   :language: python

All data sources and filters are defined as method calls on the
``Recipe`` (any hyphen is replaced by an underscore):

So the ``grib`` source is defined as ``Recipe.grib(...)`` and the
``clip`` filter as ``Recipe.clip(...)``.

Source and filter methods can be combined together and assigned to
``Recipe.input``.

Use the pipe operator ``|`` to chain sources and filters:

.. literalinclude:: code/using-python-5.py
   :language: python

Use the ampersand operator ``&`` to combine multiple inputs:

.. literalinclude:: code/using-python-6.py
   :language: python

And you can combine both operators:

.. literalinclude:: code/using-python-7.py
   :language: python

To generate the YAML configuration, call the ``dump()`` method:

.. literalinclude:: code/using-python-8.py
   :language: python

Which will output:

.. literalinclude:: yaml/using-python-1.yaml
   :language: yaml
