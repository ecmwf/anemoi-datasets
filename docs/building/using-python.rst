##############################
 Using Python defined recipes
##############################

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

So the :ref:`grib <grib-source>` source is defined as
``Recipe.grib(...)`` and the :ref:`clip <anemoi-transform:clip-filter>`
filter as ``Recipe.clip(...)``.

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

To generate the YAML configuration, call the ``Recipe.dump()`` method:

.. literalinclude:: code/using-python-8.py
   :language: python

Which will output:

.. literalinclude:: yaml/using-python-1.yaml
   :language: yaml

Sometimes you need to refer to part of the input in a source or a
filter, such as when using the :ref:`forcing_variables` source.

You can do this by assigning the result of a source or filter to a
variable, and use that variable later in the recipe.

.. literalinclude:: code/using-python-9.py
   :language: python

Or you can assigning the result of a source or filter to a variable
using the walrus operator ``:=`` to both assign and use the variable in
the same expression:

.. literalinclude:: code/using-python-10.py
   :language: python

Finally, if you need different inputs for different dates, you can use
the ``Recipe.concat()`` method, which takes a dictionary mapping dates
to inputs:

.. literalinclude:: code/using-python-11.py
   :language: python

Note that the dates can also be :class:`datetime.datetime` objects and
the frequency can be a :class:`datetime.timedelta` object.

.. note::

   To get you started quickly, you can use the :ref:`anemoi-datasets
   recipe --python recipe.yaml <recipe_command>` to transform an
   existing YAML recipe into a Python script.

Below is the complete example. It uses the :ref:`mars-source` and
:ref:`accumulations-source` source to get data from the ECMWF's MARS
archive. In addition, it uses :ref:`lnsp-to-sp
<anemoi-transform:lnsp-to-sp-filter>` to convert the logarithm of the
surface pressure to the surface pressure, :ref:`snow-cover
<anemoi-transform:snow-cover-filter>` to compute the snow cover from the
snow depth and snow density and :ref:`apply-mask
<anemoi-transform:apply-mask-filter>` to replace zeros with `NaNs`.

.. literalinclude:: code/using-python-12.py
   :language: python
