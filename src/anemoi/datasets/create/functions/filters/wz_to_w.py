# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict

from earthkit.data.indexing.fieldlist import FieldArray


class NewDataField:
    def __init__(self, field, data, new_name):
        self.field = field
        self.data = data
        self.new_name = new_name

    def to_numpy(self, *args, **kwargs):
        return self.data

    def metadata(self, key=None, **kwargs):
        if key is None:
            return self.field.metadata(**kwargs)

        value = self.field.metadata(key, **kwargs)
        if key == "param":
            return self.new_name
        return value

    def __getattr__(self, name):
        return getattr(self.field, name)


def execute(context, input, wz, t, w="w"):
    """Convert geometric vertical velocity (m/s) to vertical velocity (Pa / s)"""
    result = FieldArray()

    params = (wz, t)
    pairs = defaultdict(dict)

    for f in input:
        key = f.metadata(namespace="mars")
        param = key.pop("param")
        if param in params:
            key = tuple(key.items())

            if param in pairs[key]:
                raise ValueError(f"Duplicate field {param} for {key}")

            pairs[key][param] = f
            if param == t:
                result.append(f)
        else:
            result.append(f)

    for keys, values in pairs.items():

        if len(values) != 2:
            raise ValueError("Missing fields")

        wz_pl = values[wz].to_numpy(flatten=True)
        t_pl = values[t].to_numpy(flatten=True)
        pressure = keys[4][1] * 100  # TODO: REMOVE HARDCODED INDICES

        w_pl = wz_to_w(wz_pl, t_pl, pressure)
        result.append(NewDataField(values[wz], w_pl, w))

    return result


def wz_to_w(wz, t, pressure):
    g = 9.81
    Rd = 287.058

    return -wz * g * pressure / (t * Rd)
