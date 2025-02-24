# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import json
import logging
import os
from typing import Union

import zarr

LOG = logging.getLogger(__name__)


def fix_order_by(order_by: Union[dict, list]) -> list[dict]:
    """Fix the order_by attribute to ensure it is a list of dictionaries.

    Parameters
    ----------
    order_by : dict or list
        The order_by attribute to fix.

    Returns
    -------
    list[dict]
        The fixed order_by attribute.
    """
    if isinstance(order_by, list):
        return order_by

    assert isinstance(order_by, dict), order_by
    assert len(order_by) <= 3, order_by
    lst = []
    lst.append({"valid_datetime": order_by["valid_datetime"]})
    lst.append({"param_level": order_by["param_level"]})
    lst.append({"number": order_by["number"]})
    return lst


def fix_history(history: list[dict]) -> list[dict]:
    """Fix the history attribute by removing specific actions.

    Parameters
    ----------
    history : list[dict]
        The history attribute to fix.

    Returns
    -------
    list[dict]
        The fixed history attribute.
    """
    new = history
    new = [d for d in new if d.get("action") != "loading_data_start"]
    new = [d for d in new if d.get("action") != "loading_data_end"]
    return new


def fix_provenance(provenance: dict) -> dict:
    """Fix the provenance attribute by adding missing fields and removing unnecessary ones.

    Parameters
    ----------
    provenance : dict
        The provenance attribute to fix.

    Returns
    -------
    dict
        The fixed provenance attribute.
    """
    if "python" not in provenance:
        provenance["python"] = provenance["platform"]["python_version"]

    for q in (
        "args",
        "config_paths",
        "executable",
        "gpus",
        "platform",
        "python_path",
        "assets",
    ):
        if q in provenance:
            del provenance[q]

    for k, v in list(provenance["module_versions"].items()):
        if v.startswith("<"):
            del provenance["module_versions"][k]
        if v.startswith("/"):
            provenance["module_versions"][k] = os.path.join("...", os.path.basename(v))

    for k, v in list(provenance["git_versions"].items()):
        LOG.debug(k, v)
        modified_files = v["git"].get("modified_files", [])
        untracked_files = v["git"].get("untracked_files", [])
        if not isinstance(modified_files, int):
            modified_files = len(modified_files)
        if not isinstance(untracked_files, int):
            untracked_files = len(untracked_files)
        provenance["git_versions"][k] = dict(
            git={
                "sha1": v["git"]["sha1"],
                "modified_files": modified_files,
                "untracked_files": untracked_files,
            }
        )

    LOG.debug(json.dumps(provenance, indent=2))
    # assert False
    return provenance


def apply_patch(path: str, verbose: bool = True, dry_run: bool = False) -> None:
    """Apply a patch to the dataset at the given path.

    Parameters
    ----------
    path : str
        The path to the dataset.
    verbose : bool, optional
        Whether to log detailed information. Defaults to True.
    dry_run : bool, optional
        If True, do not actually apply the patch. Defaults to False.
    """
    LOG.debug("====================")
    LOG.debug(f"Patching {path}")
    LOG.debug("====================")

    try:
        attrs = zarr.open(path, mode="r").attrs.asdict()
    except zarr.errors.PathNotFoundError as e:
        LOG.error(f"Failed to open {path}")
        LOG.error(e)
        exit(0)

    FIXES = {
        "history": fix_history,
        "provenance_load": fix_provenance,
        "provenance_statistics": fix_provenance,
        "order_by": fix_order_by,
    }
    REMOVE = ["_create_yaml_config"]

    before = json.dumps(attrs, sort_keys=True)

    fixed_attrs = {}
    for k, v in attrs.items():
        v = attrs[k]
        if k in REMOVE:
            LOG.info(f"✅ Remove {k}")
            continue

        if k not in FIXES:
            assert not k.startswith("provenance"), f"[{k}]"
            LOG.debug(f"✅ Don't fix {k}")
            fixed_attrs[k] = v
            continue

        new_v = FIXES[k](v)
        if json.dumps(new_v, sort_keys=True) != json.dumps(v, sort_keys=True):
            LOG.info(f"✅ Fix {k}")
            if verbose:
                LOG.info(f"  Before : {k}= {v}")
                LOG.info(f"  After  : {k}= {new_v}")
        else:
            LOG.debug(f"✅ Unchanged {k}")
        fixed_attrs[k] = new_v

    if dry_run:
        return
    z = zarr.open(path, mode="r+")

    for k in list(z.attrs.keys()):
        if k not in fixed_attrs:
            del z.attrs[k]
    for k, v in fixed_attrs.items():
        z.attrs[k] = v

    after = json.dumps(z.attrs.asdict(), sort_keys=True)
    if before != after:
        LOG.info("Dataset changed by patch")

    assert json.dumps(z.attrs.asdict(), sort_keys=True) == json.dumps(fixed_attrs, sort_keys=True)
