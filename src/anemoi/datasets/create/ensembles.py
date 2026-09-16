# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""The ensemble-member axis of the output cube.

The member of a field is read from the ``ensemble.member`` *component*, never
from the raw ``metadata.number`` GRIB key: a field rebuilt by ``field.set()``
(every accumulation, forcing, filter or otherwise computed field) keeps its
components but loses access to its raw metadata, so ``metadata.number`` reads
back as missing there and all the members of such a variable would collapse
onto member 0.

earthkit-data stores the member as a *string* (and leaves it unset for
deterministic data), so the members of a 50-member ensemble would be ordered
1, 10, 11, ..., 2, 20, ... on the ensemble axis.  :func:`ensemble_order` gives
``to_cube`` the numerical order explicitly instead; :data:`ENSEMBLE_PATCH`
declares the same normalisation the earthkit way (and covers the values the
cube reports back), but note that earthkit-data 1.0.2 silently ignores a
``patch`` in ``unique()`` and in ``order_by()`` -- hence the explicit order.
"""

from typing import Any

# Component path of the ensemble member, used as the last `order_by` key of
# every layout's cube.
ENSEMBLE_KEY: str = "ensemble.member"


def normalise_ensemble_member(value: Any) -> int:
    """Normalise an ``ensemble.member`` value to the number used on the ensemble axis.

    Parameters
    ----------
    value : Any
        The raw component value: a string or integer member, or ``None`` for
        deterministic data.

    Returns
    -------
    int
        The member number, 0 when the field carries no ensemble information.
    """
    if value is None:
        return 0
    return int(value)


# Patch applied to the ensemble key of the cube. A module-level function rather
# than a lambda or a dict so that the patched remapping stays picklable.
ENSEMBLE_PATCH: dict[str, Any] = {ENSEMBLE_KEY: normalise_ensemble_member}


def ensemble_order(datasource: Any) -> list[int] | None:
    """The members of *datasource* in the order they should sit on the ensemble axis.

    Parameters
    ----------
    datasource : Any
        The fields of one group, as handed to ``to_cube``.

    Returns
    -------
    list of int or None
        The members, ordered as numbers; ``None`` when the order is either
        irrelevant (fewer than two members) or cannot be established (a field
        with no ensemble information, a datasource that is not a list of
        fields), in which case the default ordering is left in place.
    """
    try:
        members = {field.get(ENSEMBLE_KEY, default=None) for field in datasource}
    except (AttributeError, TypeError):
        # Not a list of fields (e.g. the dataframe of a tabular recipe); let
        # the caller report that in its own terms.
        return None

    if len(members) < 2 or None in members:
        return None

    # Both spellings of each value are recognised by earthkit's ordering, so
    # integers order the string members the fields actually carry.
    return sorted(normalise_ensemble_member(m) for m in members)


def ensemble_member(field: Any) -> int:
    """Return the ensemble member of a field.

    Parameters
    ----------
    field : Any
        An earthkit-data-like field.

    Returns
    -------
    int
        The member number, 0 when the field carries no ensemble information.
    """
    return normalise_ensemble_member(field.get(ENSEMBLE_KEY, default=None))
