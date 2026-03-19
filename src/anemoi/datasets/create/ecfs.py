# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import hashlib
import logging
import os
import subprocess
import tempfile

TMPDIR = None
LOG = logging.getLogger(__name__)


def get_ecfs_file(path: str) -> str:
    global TMPDIR
    if TMPDIR is None:
        TMPDIR = tempfile.mkdtemp()

    _, ext = os.path.splitext(path)
    local_name = os.path.join(TMPDIR, hashlib.sha1(path.encode()).hexdigest() + ext)
    LOG.info(f"Calling ecp {path} {local_name}")
    subprocess.check_call(["ecp", path, local_name])
    return local_name
