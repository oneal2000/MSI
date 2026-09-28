"""Cross-host writer ownership on the cluster's shared filesystem.

This filesystem does not propagate flock between hosts. Atomic mkdir does.
An uncleanly terminated owner leaves a claim for explicit operator inspection;
workers must not infer that another host's PID is dead from a local PID lookup.
"""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time


@contextmanager
def claim(path, *, wait=False):
    path = Path(path)
    owner = {'host': socket.gethostname(), 'pid': os.getpid(), 'started': time.time()}
    while True:
        try:
            path.mkdir()
            break
        except FileExistsError as error:
            if not wait:
                raise BlockingIOError(f'writer already owns {path}') from error
            time.sleep(10)
    try:
        (path / 'owner.json').write_text(json.dumps(owner))
        yield
    finally:
        owner_file = path / 'owner.json'
        if owner_file.exists() and json.loads(owner_file.read_text()) == owner:
            owner_file.unlink()
            path.rmdir()
