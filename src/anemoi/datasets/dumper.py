# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import io
import logging

import ruamel.yaml

LOG = logging.getLogger(__name__)


def represent_date(dumper, data):

    if isinstance(data, datetime.datetime):
        if data.tzinfo is None:
            data = data.replace(tzinfo=datetime.timezone.utc)
        data = data.astimezone(datetime.timezone.utc)
        iso_str = data.replace(tzinfo=None).isoformat(timespec="seconds") + "Z"
    else:
        iso_str = data.isoformat()

    return dumper.represent_scalar("tag:yaml.org,2002:timestamp", iso_str)


# --- Represent multiline strings with | style ---
def represent_multiline_str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data.strip(), style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


# --- Represent short lists inline (flow style) ---
def represent_inline_list(dumper, data):

    if not all(isinstance(i, (str, int, float, bool, type(None))) for i in data):
        return dumper.represent_sequence("tag:yaml.org,2002:seq", data)

    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def yaml_dump(obj, order=None, stream=None, **kwargs):

    if order:

        def _ordering(k):
            return order.index(k) if k in order else len(order)

        obj = {k: v for k, v in sorted(obj.items(), key=lambda item: _ordering(item[0]))}

    yaml = ruamel.yaml.YAML()
    yaml.width = 120  # wrap long flow sequences

    yaml.Representer.add_representer(datetime.date, represent_date)
    yaml.Representer.add_representer(datetime.datetime, represent_date)
    yaml.Representer.add_representer(str, represent_multiline_str)
    yaml.Representer.add_representer(list, represent_inline_list)

    data = ruamel.yaml.comments.CommentedMap()
    for i, (k, v) in enumerate(obj.items()):
        data[k] = v
        if i > 0:
            data.yaml_set_comment_before_after_key(key=k, before="\n")

    if stream:
        yaml.dump(data, stream=stream, **kwargs)

    stream = io.StringIO()
    yaml.dump(data, stream=stream, **kwargs)
    return stream.getvalue()
