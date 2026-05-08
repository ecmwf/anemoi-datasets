.. _naming-conventions:

############################
 Dataset naming conventions
############################

A dataset name is a string used to identify a dataset. It is designed to
be human-readable and is *not* designed to be parsed and split into
parts.

To ensure consistency, a dataset name should follow the following rules:

   -  All lower case.
   -  Only letters, numbers, and dashes ``-`` are allowed.
   -  No underscores ``_``, no dots ``.``, no uppercase letters, and no
      other special characters (``@``, ``#``, ``*``, etc.).

Additionally, a dataset name is built from different parts joined with
``-`` as follows (each part can contain additional ``-``).  The exact
form depends on the dataset's layout:

.. code::

   # Gridded layout (single date frequency)
   purpose-content-source-resolution-start-year-end-year-frequency-version[-extra-str]

   # Trajectory layout (two frequencies: between base dates and between forecast steps)
   purpose-content-source-resolution-start-year-end-year-date-frequency-step-frequency-version[-extra-str]

   # Tabular layout (no frequency; resolution is optional)
   purpose-content-source[-resolution]-start-year-end-year-version[-extra-str]

.. note::

   This is the current naming convention for datasets in the Anemoi
   registry. It will need to be updated and adapted as more datasets are
   added. The part **purpose** is especially difficult to define for
   some datasets and may be revisited.

The tables below provide more details and some examples.

.. list-table:: Dataset naming conventions
   :widths: 20 80
   :header-rows: 1

   -  -  Component
      -  Description

   -  -  **purpose**

      -  Can be `aifs` because the data is used to train the AIFS model.
         It is also sometimes `metno` for data from the Norwegian
         Meteorological Institute. This definition may need to be
         revisited.

   -  -  **content**

      -  The content of the dataset CAN have four parts, such as:
         *class-type-stream-expver*

         -  **class**: od Operational archive (*class* is a MARS
            keyword)
         -  **type**: an Analysis (*type* is a MARS keyword)
         -  **stream**: oper Atmospheric model (*stream* is a MARS
            keyword)
         -  **expver**: 0001 (operational model)

   -  -  **source**
      -  mars (when data is from MARS), could be *opendap* or other.

   -  -  **resolution**

      -  o96 (could be : n320, 0p2 for 0.2 degree, 1km, 2km).  The
         resolution token must start with a digit (``0``-``9``), ``o``
         or ``n``.  It is **mandatory** for gridded and trajectory
         layouts and **optional** for tabular layouts (since station
         observation datasets often have no meaningful spatial
         resolution).

   -  -  **start-year**
      -  1979 if the first validity time is in 1979.

   -  -  **end-year**

      -  2020 if the last validity time is in 2020. Notice that if the
         dataset is from 18.04.2020 to 19.07.2020, the start-year and
         end-year are both 2020. For instance in
         aifs-od-an-oper-0001-mars-o96-2020-2020-6h-v5

   -  -  **frequency**

      -  1h (could be : 6h, 10m for 10 minutes).  The frequency token
         is **mandatory** for gridded layouts and **absent** for
         tabular layouts.

   -  -  **date-frequency**, **step-frequency**

      -  Trajectory datasets carry **two** frequencies, in this order:
         the *date frequency* is the interval between consecutive base
         dates (forecast initialisation times) and the *step frequency*
         is the interval between consecutive forecast steps within one
         trajectory.  For example
         ``aifs-od-fc-oper-0001-mars-o96-2016-2025-18h-1h-v3`` describes
         a trajectory dataset with one forecast every 18 h and hourly
         steps within each forecast.

   -  -  **version**

      -  This is the version of the content of the dataset, e.g. which
         variables, levels, etc. This is not the version of the format.
         There must be a "v" before the version number. The "v" is not
         part of the version number. For instance ...-v5 is the fifth
         version of the content of the dataset.

   -  -  **extra-str**

      -  Experimental datasets can have additional text in the name.
         This extra string can contain additional `-`. It provides
         additional information about the content of the dataset.

.. list-table:: Examples — gridded
   :widths: 100

   -  -  aifs-od-an-oper-0001-mars-o96-1979-2022-1h-v5
   -  -  aifs-ea-an-oper-0001-mars-o96-1979-2022-6h-v6
   -  -  aifs-ea-an-enda-0001-mars-o96-1979-2022-6h-v6-recentered-on-oper
   -  -  aifs-ea-an-oper-0001-mars-n320-1979-2022-6h-v4
   -  -  inca-an-oper-0001-gridefix-1km-2023-2024-10m-v1

.. list-table:: Examples — trajectories
   :widths: 100

   -  -  aifs-od-fc-oper-0001-mars-o96-2016-2025-18h-1h-v3

.. list-table:: Examples — tabular
   :widths: 100

   -  -  aifs-od-fc-oper-0001-mars-o96-2016-2025-v3
   -  -  observations-ea-ofb-0001-1979-2023-v2-combined-aircraft-from-dop
   -  -  dop-ea-ofb-0001-1979-2023-v2-combined-aircraft
   -  -  dop-ea-ofb-0001-2km-1979-2023-v2-combined-aircraft

`Anemoi Naming Conventions
<https://anemoi-registry.readthedocs.io/en/latest/naming-conventions.html>`_
