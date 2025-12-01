import logging
import threading

import fasteners
import rich

LOG = logging.getLogger(__name__)


class Locking:

    def __init__(self, path: str):
        rich.print(f"[green]Using locking file at {path}[/green]")
        self.flock = fasteners.InterProcessLock(path, logger=LOG)
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
