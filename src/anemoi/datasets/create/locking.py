# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import threading

import fasteners
import rich


class Locking:

    def __init__(self, path: str):
        self.flock = fasteners.InterProcessLock(path)
        self.tlock = threading.RLock()
        self._acquired = 0

    def acquire(self):
        self.tlock.acquire()
        if self._acquired == 0:
            self.flock.acquire()
        self._acquired += 1

    def release(self):
        self._acquired -= 1
        if self._acquired == 0:
            self.flock.release()
        self.tlock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


if __name__ == "__main__":
    lock = Locking("./test.lock")
    with lock:
        rich.print("Locked")
        import time

        with lock:
            time.sleep(5)
        rich.print("Unlocked")
