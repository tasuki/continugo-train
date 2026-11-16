"""Append-only stage durations; monotonic clocks exclude downtime between resumes."""

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

ENVIRONMENT = "CONTINUGO_TIMING_FILE"


def journal(root, number):
    return Path(root) / f"iteration-{number:04d}" / "timing.jsonl"


@contextmanager
def measure(path, phase):
    """Retain failed/interrupted work too; never let timing I/O mask a stage error."""
    started = time.perf_counter()
    completed = False
    try:
        yield
        completed = True
    finally:
        event = {
            "phase": phase,
            "seconds": time.perf_counter() - started,
            "ended_at": time.time(),
            "completed": completed,
        }
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # One append write allows the evaluator child to share this journal.
            # Leading newline isolates a partial append left by a killed writer.
            data = ("\n" + json.dumps(event, sort_keys=True) + "\n").encode()
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, data)
            finally:
                os.close(fd)
        except OSError as error:
            print(f"WARNING: timing not recorded: {error}", flush=True)
