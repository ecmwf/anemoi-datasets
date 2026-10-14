# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared machinery for sources that reduce archived fields over a time window.

`accumulate` and the time reductions (`average`, `minimum`, `maximum`) are built on
this package. The code is split by kind of source data, not by operation:

- fields that each *span* an interval are covered by subwindows (`covering`,
  `subwindows`), each rebuilt by a signed sum, and reduced by `ranges.reduce_ranges`;
- instantaneous fields are sampled inside the window, by the loop in
  `reduce/source.py`.

Both reduce with one of four `operations` and ask one table, `check.check`, whether
the reduction suits what the archive stores.
"""
