# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from earthkit.data.core.temporary import temp_file
from earthkit.data.readers.grib.output import new_grib_output

LOG = logging.getLogger(__name__)

CLIP_VARIABLES = (
    "q",
    "cp",
    "lsp",
    "tp",
    "sf",
    "swl4",
    "swl3",
    "swl2",
    "swl1",
)

SKIP = ("class", "stream", "type", "number", "expver", "_leg_number", "anoffset", "time", "date", "step")


def check_compatible(
    f1: Any, f2: Any, centre_field_as_mars: Dict[str, Any], ensemble_field_as_mars: Dict[str, Any]
) -> None:
    """Check if two fields are compatible.

    Parameters
    ----------
    f1 : Any
        The first field.
    f2 : Any
        The second field.
    centre_field_as_mars : Dict[str, Any]
        Metadata of the centre field.
    ensemble_field_as_mars : Dict[str, Any]
        Metadata of the ensemble field.
    """
    assert f1.mars_grid == f2.mars_grid, (f1.mars_grid, f2.mars_grid)
    assert f1.mars_area == f2.mars_area, (f1.mars_area, f2.mars_area)
    assert f1.shape == f2.shape, (f1.shape, f2.shape)

    # Not in *_as_mars
    assert f1.metadata("valid_datetime") == f2.metadata("valid_datetime"), (
        f1.metadata("valid_datetime"),
        f2.metadata("valid_datetime"),
    )

    for k in set(centre_field_as_mars.keys()) | set(ensemble_field_as_mars.keys()):
        if k in SKIP:
            continue
        assert centre_field_as_mars[k] == ensemble_field_as_mars[k], (
            k,
            centre_field_as_mars[k],
            ensemble_field_as_mars[k],
        )


def recentre(
    *,
    members: Any,
    centre: Any,
    clip_variables: Tuple[str, ...] = CLIP_VARIABLES,
    alpha: float = 1.0,
    output: Optional[str] = None,
) -> Any:
    """Recentre ensemble members around the centre field.

    Parameters
    ----------
    members : Any
        The ensemble members.
    centre : Any
        The centre field.
    clip_variables : Tuple[str, ...], optional
        Variables to clip. Defaults to CLIP_VARIABLES.
    alpha : float, optional
        Scaling factor. Defaults to 1.0.
    output : Optional[str], optional
        Output path. Defaults to None.

    Returns
    -------
    Any
        The recentred dataset or output path.
    """
    keys = ["param", "level", "valid_datetime", "date", "time", "step", "number"]

    number_list = members.unique_values("number", progress_bar=False)["number"]
    n_numbers = len(number_list)

    assert None not in number_list

    LOG.info("Ordering fields")
    members = members.order_by(*keys)
    centre = centre.order_by(*keys)
    LOG.info("Done")

    if len(centre) * n_numbers != len(members):
        LOG.error("%s %s %s", len(centre), n_numbers, len(members))
        for f in members:
            LOG.error("Member: %r", f)
        for f in centre:
            LOG.error("centre: %r", f)
        raise ValueError(f"Inconsistent number of fields: {len(centre)} * {n_numbers} != {len(members)}")

    if output is None:
        # prepare output tmp file so we can read it back
        tmp = temp_file()
        path = tmp.path
    else:
        tmp = None
        path = output

    out = new_grib_output(path)

    seen = set()

    for i, centre_field in enumerate(centre):
        param = centre_field.metadata("param")
        centre_field_as_mars = centre_field.metadata(namespace="mars")

        # load the centre field
        centre_np = centre_field.to_numpy()

        # load the ensemble fields and compute the mean
        members_np = np.zeros((n_numbers, *centre_np.shape))

        for j in range(n_numbers):
            ensemble_field = members[i * n_numbers + j]
            ensemble_field_as_mars = ensemble_field.metadata(namespace="mars")
            check_compatible(
                centre_field,
                ensemble_field,
                centre_field_as_mars,
                ensemble_field_as_mars,
            )
            members_np[j] = ensemble_field.to_numpy()

            ensemble_field_as_mars = tuple(sorted(ensemble_field_as_mars.items()))
            assert ensemble_field_as_mars not in seen, ensemble_field_as_mars
            seen.add(ensemble_field_as_mars)

        # cmin=np.amin(centre_np)
        # emin=np.amin(members_np)

        # if cmin < 0 and emin >= 0:
        #     LOG.warning(f"Negative values in {param} cmin={cmin} emin={emin}")
        #     LOG.warning(f"centre: {centre_field_as_mars}")

        mean_np = members_np.mean(axis=0)

        for j in range(n_numbers):
            template = members[i * n_numbers + j]
            e = members_np[j]
            m = mean_np
            c = centre_np

            assert e.shape == c.shape == m.shape, (e.shape, c.shape, m.shape)

            x = c + (e - m) * alpha

            if param in clip_variables:
                # LOG.warning(f"Clipping {param} to be positive")
                x = np.maximum(x, 0)

            assert x.shape == e.shape, (x.shape, e.shape)

            out.write(x, template=template)
            template = None

    assert len(seen) == len(members), (len(seen), len(members))

    out.close()

    if output is not None:
        return path

    from earthkit.data import from_source

    ds = from_source("file", path)

    # save a reference to the tmp file so it is deleted
    # only when the dataset is not used anymore
    ds._tmp = tmp

    assert len(ds) == len(members), (len(ds), len(members))

    return ds
