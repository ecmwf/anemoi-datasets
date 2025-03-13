# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
import shutil
import time

from filelock import FileLock
from filelock import Timeout

LOG = logging.getLogger(__name__)


class NoSynchroniser:
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def clean(self):
        pass


class Synchroniser:
    def __init__(self, lock_file_path, timeout=10):
        """Initialize the Synchroniser with the path to the lock file and an optional timeout.
        Parameters
        ----------
        lock_file_path
            Path to the lock file on a shared filesystem.
        timeout
            Timeout for acquiring the lock in seconds.
        """
        self.lock_file_path = lock_file_path
        self.timeout = timeout
        self.lock = FileLock(lock_file_path)

    def __enter__(self):
        """Acquire the lock when entering the context."""
        try:
            self.lock.acquire(timeout=self.timeout)
            print("Lock acquired.")
        except Timeout:
            print("Could not acquire lock, another process might be holding it.")
            raise
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Release the lock when exiting the context."""
        self.lock.release()
        print("Lock released.")

    def clean(self):
        try:
            shutil.rmtree(self.lock_file_path)
        except FileNotFoundError:
            pass


# Example usage
if __name__ == "__main__":

    def example_operation():
        print("Performing operation...")
        time.sleep(2)  # Simulate some work
        print("Operation complete.")

    lock_path = "/path/to/shared/lockfile.lock"

    # Use the Synchroniser as a context manager
    with Synchroniser(lock_path) as sync:
        example_operation()
