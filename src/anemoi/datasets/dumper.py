# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import yaml

LOG = logging.getLogger(__name__)


class MyDumper(yaml.SafeDumper):
    pass


def represent_date(dumper, data):
    if data.tzinfo is None:
        data = data.replace(tzinfo=datetime.timezone.utc)
    data = data.astimezone(datetime.timezone.utc)
    iso_str = data.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", iso_str)


# --- Represent multiline strings with | style ---
def represent_multiline_str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# --- Represent short lists inline (flow style) ---
def represent_inline_list(dumper, data):

    if not all(isinstance(i, (str, int, float, bool, type(None))) for i in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data)

    elems = [yaml.dump(i, explicit_start=False, explicit_end=False).replace("\n...\n", "") for i in data]
    lines = []
    line = []
    for e in elems:
        if sum(len(x) for x in line) + len(e) + 2 * (len(line) + 1) <= 80:
            line.append(e)
        else:
            lines.append(line)
            line = [e]

    if line:
        lines.append(line)

    if len(lines) == 1:
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

    block_lines = ["- [" + ", ".join(line) + "]" for line in lines]
    return dumper.represent_scalar("tag:yaml.org,2002:str", "\n".join(block_lines), style="|")


# Register representers
MyDumper.add_representer(datetime.date, represent_date)
MyDumper.add_representer(datetime.datetime, represent_date)
MyDumper.add_representer(str, represent_multiline_str)
MyDumper.add_representer(list, represent_inline_list)


def yaml_dump(obj, **kwargs):
    return yaml.dump(obj, Dumper=MyDumper, **kwargs)
