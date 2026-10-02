# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
import json

import numpy as np
import pytest
import zarr

from anemoi.datasets.__main__ import main

nexus_record = importlib.import_module("anemoi.datasets.commands.nexus-record")


def _dataset(path, uuid="5b0d7b2e-1111-4c6d-9e7f-0a1b2c3d4e5f"):
    z = zarr.open_group(str(path), mode="w")
    z.attrs.update({"uuid": uuid, "variables": ["2t", "10u"]})
    z.create_dataset("data", data=np.zeros((4, 2, 1, 3), dtype="f4"), chunks=(1, 2, 1, 3))
    z.create_dataset("mean", data=np.array([1.0, 2.0]))
    return path


def test_dataset_nexus_record(tmp_path) -> None:
    path = _dataset(tmp_path / "my-ds.zarr")
    record = nexus_record.dataset_nexus_record(str(path) + "/", {"owner": "alice", "projects": ["MLP"]})
    assert record["name"] == "my-ds" and record["uuid"] == "5b0d7b2e-1111-4c6d-9e7f-0a1b2c3d4e5f"
    assert record["owner"] == "alice" and record["projects"] == ["MLP"]
    meta = record["metadata"]
    assert meta["variables"] == ["2t", "10u"] and meta["statistics"] == {"mean": [1.0, 2.0]}
    assert meta["shape"] == [4, 2, 1, 3] and meta["dtype"] == "float32" and meta["chunks"] == [1, 2, 1, 3]
    with pytest.raises(ValueError, match="must end in .zarr"):
        nexus_record.dataset_nexus_record(str(tmp_path / "x"))


def test_nexus_record_command(tmp_path, capsys, monkeypatch) -> None:
    path = _dataset(tmp_path / "my-ds.zarr")
    attrs = tmp_path / "attrs.yaml"
    attrs.write_text("project: MLP\nlicences: [CC-BY-4.0]\n")
    out = tmp_path / "record.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "anemoi-datasets",
            "nexus-record",
            str(path),
            f"@{attrs}",
            "--owner",
            "bob",
            "-o",
            str(out),
        ],
    )
    with pytest.raises(SystemExit) as exit:
        main()
    assert exit.value.code in (0, None)
    record = json.loads(out.read_text())
    assert record["projects"] == ["MLP"] and record["licenses"] == ["CC-BY-4.0"] and record["owner"] == "bob"
    assert record["name"] == "my-ds" and record["metadata"]["shape"] == [4, 2, 1, 3]
