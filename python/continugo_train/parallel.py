"""CPU budgeting shared by independent, single-threaded jobs."""

import os


def worker_count(jobs):
    """Use at most one worker per available CPU and per runnable job.

    Respect process CPU affinity rather than the host's total CPU count.
    Return zero for an empty batch; no pool is needed for zero or one job.
    """
    if jobs < 0:
        raise ValueError("job count must be nonnegative")
    if hasattr(os, "process_cpu_count"):
        cpus = os.process_cpu_count()
    elif hasattr(os, "sched_getaffinity"):
        cpus = len(os.sched_getaffinity(0))
    else:
        cpus = os.cpu_count()
    return min(jobs, max(1, cpus or 1))
