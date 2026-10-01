# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Shared machinery for sources that reduce archived fields over a time window.

`accumulate` and the time reductions (`average`, `minimum`, `maximum`) are thin
layers on top of this package: they differ in the operation applied over the
window, not in how the window is resolved, retrieved or grouped.

Nothing here knows a recipe key.
"""
