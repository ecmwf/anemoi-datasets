import logging
from copy import deepcopy

from anemoi.datasets import open_dataset

LOG = logging.getLogger(__name__)


def _lookup(top, paths, default):
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for path in paths:
        obj = top
        for p in path.split("."):
            obj = obj.get(p)
            if obj is None:
                break
        if obj is not None:
            return obj
    return default


def record_to_display(record, mapping, defaults={}):
    display = dict(**record.get("display", {}))
    for k, v in mapping.items():
        display[k] = _lookup(record, v, defaults.get(k, "-"))
    return display


def to_by(values):
    try:
        diff = int(values[1]) - int(values[0])
        for i, s in enumerate(values[2:]):
            if int(values[i + 2]) - int(values[i + 1]) != diff:
                return values

    except ValueError:
        return values

    if diff == 1:
        return [" ".join(str(x) for x in [values[0], "to", values[-1]])]
    else:
        return [" ".join(str(x) for x in [values[0], "to", values[-1], "by", diff])]


def build_forcing_and_variables(record):
    metadata = record["metadata"]
    param_level = metadata["data_request"]["param_level"]
    display = {}
    seen = set()
    for levtype, values in param_level.items():
        display[levtype] = []
        levels = []
        params = []
        for value in values:
            if isinstance(value, list):
                params.append(value[0])
                levels.append(value[1])
                value = "_".join(str(x) for x in value)
                seen.add(value)
            else:
                value = str(value)
                seen.add(value)
                display[levtype].append(value)

        if levels:
            levels = to_by(sorted(set(levels)))
            levels = ", ".join(str(x) for x in levels)
            params = ", ".join(str(x) for x in sorted(set(params)))
            display[levtype] = f"{params} on levels: {levels}"
        else:
            display[levtype] = ", ".join(display[levtype])

    variables = sorted(set(metadata["variables"]))
    forcings = []

    # Keep the order

    for v in variables:
        if v not in seen:
            forcings.append(v)
    display["forcings"] = ", ".join(sorted(forcings))
    return display


def create_catalogue_record(path):
    import os

    import zarr

    z = zarr.open(path)
    ds = open_dataset(path)
    name, _ = os.path.splitext(os.path.basename(path))

    record = {
        "name": name,
        "uuid": z.attrs.get("uuid"),
        "metadata": z.attrs.asdict(),
        "display": {},
        "statistics": ds.statistics,
    }

    display = record["display"]

    display["shape"] = z.data.shape

    mapping = {
        "name": "name",
        "description": "metadata.description",
        "format": "metadata.version",
        "size": "metadata.total_size",
        "files": "metadata.total_number_of_files",
        "uuid": "metadata.uuid",
        "created": "metadata.latest_write_timestamp",
        "start_date": "metadata.start_date",
        "end_date": "metadata.end_date",
        "resolution": "metadata.resolution",
        "frequency": "metadata.frequency",
        "statistics_start_date": "metadata.statistics_start_date",
        "statistics_end_date": "metadata.statistics_end_date",
        "attribution": "metadata.attribution",
        "licence": "metadata.licence",
    }

    display.update(record_to_display(record, mapping))

    display.update(build_forcing_and_variables(record))

    display["dtype"] = ds.dtype
    display["chunks"] = ds.chunks

    record = prepare_serialising_catalogue_record(record)

    # check
    _ = finish_deserialisation_catalogue_record(record)
    assert _["display"]["dtype"] == record["display"]["dtype"]
    for k in record["statistics"]:
        assert (_["statistics"][k] == record["statistics"][k]).all()

    return record


def prepare_serialising_catalogue_record(record):
    new = deepcopy(record)

    new["statistics"] = {k: v.tolist() for k, v in record["statistics"].items()}
    new["display"]["dtype"] = str(record["display"]["dtype"])

    return new


def finish_deserialisation_catalogue_record(record):
    import numpy as np

    new = deepcopy(record)

    new["statistics"] = {k: np.array(v) for k, v in record["statistics"].items()}
    new["display"]["dtype"] = np.dtype(record["display"]["dtype"])
    return new


if __name__ == "__main__":
    import sys

    print(create_catalogue_record(sys.argv[1]))
