"""Bounded, game-level parallelism for CPU neural reports."""

import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from . import archive, nn_view
from .parallel import worker_count


def _initialize_worker():
    # Each game also owns a feature subprocess. Avoid nested torch thread pools.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def _render(job):
    sgf, output, info, checkpoints, binary = job
    output = Path(output)
    # Never leave a partial file that refresh() would mistake for a finished report.
    with tempfile.TemporaryDirectory(prefix=".nn-", dir=output.parent) as temporary:
        draft = Path(temporary) / "report.html"
        nn_view.generate_rated(sgf, draft, info, checkpoints, binary=binary)
        archive.seal_file(draft)
        os.replace(draft, output)
        archive.sync_directory(output.parent)


def generate(jobs):
    """Render independent games using the process's available CPU budget.

    Use spawn rather than fork: callers have already initialized PyTorch.
    Each single-threaded worker handles one game at a time.
    Errors propagate before the caller can mark evaluation complete; completed
    reports remain available for a retry.
    """
    jobs = list(jobs)
    if not jobs:
        return
    workers = worker_count(len(jobs))
    started = time.monotonic()
    print(f"NN viewers: {len(jobs)} games, {workers} workers", file=sys.stderr, flush=True)
    if workers == 1:
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            for job in jobs:
                _render(job)
        finally:
            torch.set_num_threads(threads)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker,
        ) as pool:
            # Consume results so worker failures are never silently ignored.
            for _ in pool.map(_render, jobs, chunksize=1):
                pass
    print(
        f"NN viewers: completed {len(jobs)} games in {time.monotonic() - started:.2f}s",
        file=sys.stderr,
        flush=True,
    )
