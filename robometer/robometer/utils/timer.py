import time
from contextlib import contextmanager

from codetiming import Timer


@contextmanager
def timer(name: str, verbose: bool = False):
    """Context manager for timing operations."""
    start_time = time.time()
    if verbose:
        print(f"    Starting {name}...")
    try:
        yield
    finally:
        end_time = time.time()
        duration = end_time - start_time
        if verbose:
            print(f"    {name} completed in {duration:.2f}s")


@contextmanager
def _timer(name: str, timing_raw: dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last
