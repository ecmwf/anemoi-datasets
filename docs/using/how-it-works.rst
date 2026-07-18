.. _using-how-it-works:

##############
 How it works
##############

The operations described in this section of the documentation
(:ref:`subsetting <subsetting-datasets>`, :ref:`selecting
<using-selecting>`, :ref:`combining <combining-datasets>`,
:ref:`cutout <combining-datasets>`, etc.) are not independent features
bolted onto a reader. They are **composable building blocks**: every
operation takes one or more datasets as input and returns something that
is, again, a dataset. This page explains that idea at three levels of
detail.

.. note::

   These building blocks do not yet have an official name in the
   documentation. Throughout this page we refer to them as **dataset
   views** (by analogy with NumPy views: lazy objects that behave like
   the data they wrap). Other candidates are *operators*, *adapters*,
   *nodes* or *lenses* — suggestions are welcome.

***********************************
 High level: everything composes
***********************************

The single most important property of *anemoi-datasets* is that the
result of an operation behaves exactly like a dataset: it has the same
methods and attributes (``shape``, ``variables``, ``dates``,
``statistics``, ``__getitem__``, ...) and it loads its data lazily. This
is a *closure* property: operations take datasets and return datasets,
so they can be chained without limit.

Because of this, you can build a complex dataset step by step, feeding
the result of one ``open_dataset`` call into the next:

.. literalinclude:: code/compose_steps.py

Each step wraps the previous one. You do not have to materialise the
intermediate results: nothing is read from disk until you index the
final object.

The same network can be expressed as a single nested call:

.. literalinclude:: code/compose_nested.py

The operations can equally be passed as keyword arguments to a single
call, without wrapping them in a dictionary:

.. literalinclude:: code/compose_kwargs.py

Subsetting, selecting, joining, concatenating, cutting out, rescaling,
filling missing dates, ... all follow the same rule, which is why they
can be combined freely.

**********************************************
 Middle level: a network of interacting views
**********************************************

When you call ``open_dataset``, the keyword arguments are interpreted
one by one. Each recognised operation wraps its input in a new **dataset
view** and returns it, so the arguments build up a network (a tree, or
more generally a directed acyclic graph) of objects:

-  the **leaves** are the actual zarr stores on disk or in the cloud;

-  the **internal nodes** are the views created by ``start`` / ``end``,
   ``select``, ``frequency``, ``join``, ``cutout``, and so on;

-  the **root** is the object returned to you.

No data is copied when the network is built. Instead, when you access
``ds[i]``, the request travels from the root down to the leaves: each
view transforms the index and/or the data it receives from its children
and passes the result back up. A time subset shifts indices, a variable
selection picks columns, a join stacks the arrays of its children, a
cutout merges grid points, and so on. The data is read from the zarr
stores only where and when it is actually needed.

You can inspect the network with the :ref:`tree() <using-methods>`
method:

.. literalinclude:: code/compose_tree.py

For the example above, the views form a simple chain from the root (the
object returned to you) down to the leaf (the store):

.. literalinclude:: trees/compose_tree.txt

The chain is linear because every operation here wraps a single dataset.
As soon as datasets are **combined**, the tree branches. For instance,
the :ref:`join <combining-datasets>` configuration used later on this
page (two datasets combined, the second one reduced to two variables,
the whole thing subset in time and frequency) builds this network:

.. literalinclude:: trees/compose_config.txt

The leaves are always the actual stores; everything above them is a lazy
view. Reading ``ds[i]`` sends the request down through this tree and
assembles the answer on the way back up.

Combining two datasets always makes the tree branch, whatever the
operation. Two datasets covering consecutive periods can be
concatenated along time:

.. literalinclude:: code/combine_concat.py

.. literalinclude:: trees/combine_concat.txt

and a high-resolution regional dataset can be cut into a global one, so
that its grid points replace the global ones where the two overlap:

.. literalinclude:: code/combine_cutout.py

.. literalinclude:: trees/combine_cutout.txt

Because the whole processing chain is captured by the network of views,
the datasets are **metadata rich**: the :ref:`metadata()
<using-methods>` method records every operation that was applied, the
sources that were combined, the provenance of the run and the supporting
arrays (latitudes, longitudes, etc.).

.. literalinclude:: code/compose_metadata.py

This metadata is stored in the model checkpoints at training time, so
that the exact data pipeline used to train a model is always
recoverable.

*************************************************
 Why a single ``open_dataset`` with a dictionary
*************************************************

The network of views is entirely described by the arguments passed to
``open_dataset``. Because those arguments can be given as a single
(possibly deeply nested) dictionary, **the whole pipeline can be
described as data** rather than code:

.. literalinclude:: yaml/compose_config.yaml
   :language: yaml

.. literalinclude:: code/compose_open_config.py

This is the reason the package exposes a single ``open_dataset``
function driven by a dictionary, rather than a collection of separate
functions. A training framework — typically driven by YAML files through
Hydra_ — can let users describe an arbitrarily complex combination of
datasets in their configuration file, and hand that configuration
straight to ``open_dataset``. The same configuration fully and
reproducibly defines the training data.

.. _Hydra: https://hydra.cc/

********************************************
 Low level: views are created by factories
********************************************

Internally, each operation is a class that subclasses the base
``Dataset``. Views that wrap a single dataset derive from ``Forwards``
(they forward everything they do not override to their child); views
that combine several datasets derive from ``Combined``.

The classes are not hard-wired into ``open_dataset``. When an operation
is requested, it is resolved by name through a small factory mechanism
that is aware of the dataset's *layout* (gridded, tabular, trajectory).
The factory looks up the class in the layout-specific package first and
then in a shared ``common`` package. This means:

-  the same keyword (for example ``select``) can be implemented
   differently for different layouts;

-  the set of operations can be **extended** by adding a new module that
   defines the corresponding view class and, where relevant, a factory
   function — without modifying the dispatch code.

If you request an operation that is not available for a given layout,
the factory raises an explicit error rather than silently ignoring it.
