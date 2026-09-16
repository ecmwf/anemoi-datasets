.. _grib-index_source:

############
 grib-index
############

The `grib-index` source is used to read GRIB files with the help of an
index file created with the `grib-index` :ref:`command
<grib-index_command>`.

Because the index resolves each message to a file and a byte offset, only
the messages a recipe asks for are read; the files themselves are never
scanned again. On an archive of large multi-parameter files this is much
cheaper than selecting within each file.

Forecast archives and trajectories
==================================

The index records the model run as MARS-style ``date`` and ``time``, so
the source can also serve a :ref:`trajectory <layouts-trajectories>`
recipe, where each row is a ``(base date, step)`` pair:

.. literalinclude:: yaml/grib-index2.yaml
   :language: yaml

Note that the recipe contains no file paths and no ``step``. Within one
run the validity time already determines the lead time, so the row is
addressed by the run alone — which also means the archive's own ``step``
encoding never has to be guessed at. This matters in practice: eccodes
reports a step whose GRIB unit is minutes as ``"0m"`` rather than ``"0"``,
and one archive can mix the two.

Accumulations
=============

Under ``accumulate:``, each archived field is addressed by whatever the
archive provides. A field that belongs to a model run is fetched by that
run, so an archive accumulating from the start of its run works (a
``from-zero`` window becomes ``+a(base->step) - a(base->step-period)``,
and both terms are looked up by run and validity time). A field from a
validity-time-indexed archive with no run is fetched by its accumulation
length instead. Nothing in the recipe has to declare which: the covering
already knows.

For when to prefer this source over reading the same files by path, see
:ref:`grib-or-grib-index`.

See :ref:`create-grib-data` for more information.
