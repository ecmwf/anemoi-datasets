# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The recipe ``group_by:`` block: which metadata identifies one variable."""

def patch_groupby_keys(group_by: dict | None = None, *, source_name: str = "accumulate"):
    """Validate a recipe ``group_by:`` block, filling in the default.

    Shared with the time-reduction sources (``average``/``minimum``/``maximum``),
    which use the same key with the same meaning; *source_name* only names the
    caller in the error messages.
    """
    if group_by is None:
        return {"namespace": "mars", "ignore": ["date", "time", "step"]}
    else:
        namespace = group_by.get("namespace", None)
        if namespace is None:
            raise ValueError("No namespace in group_by (set namespace: mars for default)")
        if namespace != "mars":
            raise ValueError(f"Namespace {namespace} not supported, use 'mars'")
        ignore = group_by.get("ignore", [])
        for key in ["date", "time", "step"]:
            if key not in ignore:
                raise ValueError(
                    f"{source_name} group_by: '{key}' absent in ignore list {ignore}; "
                    "at least 'date', 'time', 'step' are required"
                )
        return group_by
