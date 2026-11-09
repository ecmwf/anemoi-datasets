# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Import the compatibility layer for its import side effect: under zarr2 it
# monkey-patches ``zarr.Group.create_array`` so tests can build stores using
# the zarr3 API regardless of the installed zarr version.
import anemoi.datasets.compat  # noqa: F401,E402

pytest_plugins = "anemoi.utils.testing"
