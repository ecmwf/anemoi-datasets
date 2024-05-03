#!/usr/bin/env python3
import json
import os

import zarr


def fix_order_by(order_by):
    if isinstance(order_by, list):
        return order_by

    assert isinstance(order_by, dict), order_by
    assert len(order_by) <= 3, order_by
    lst = []
    lst.append({"valid_datetime": order_by["valid_datetime"]})
    lst.append({"param_level": order_by["param_level"]})
    lst.append({"number": order_by["number"]})
    return lst


def fix_history(history):
    new = history
    new = [d for d in new if d.get("action") != "loading_data_start"]
    new = [d for d in new if d.get("action") != "loading_data_end"]
    return new


def fix_provenance(provenance):
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
        print(k, v)
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

    print(json.dumps(provenance, indent=2))
    # assert False
    return provenance


def apply_patch(path, verbose=True, dry_run=False):
    print("====================")
    print(f"Patching {path}")
    print("====================")

    try:
        attrs = zarr.open(path, mode="r").attrs.asdict()
    except zarr.errors.PathNotFoundError as e:
        print(f"Failed to open {path}")
        print(e)
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
            print(f"✅ Remove {k}")
            continue

        if k not in FIXES:
            assert not k.startswith("provenance"), f"[{k}]"
            print(f"✅ Don't fix {k}")
            fixed_attrs[k] = v
            continue

        new_v = FIXES[k](v)
        if json.dumps(new_v, sort_keys=True) != json.dumps(v, sort_keys=True):
            print(f"✅ Fix {k}")
            if verbose:
                print(f"  Before : {k}= {v}")
                print(f"  After  : {k}= {new_v}")
        else:
            print(f"✅ Unchanged {k}")
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
        print("CHANGED")

    assert json.dumps(z.attrs.asdict(), sort_keys=True) == json.dumps(fixed_attrs, sort_keys=True)
