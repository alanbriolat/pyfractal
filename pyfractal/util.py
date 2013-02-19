import time

import numpy as np


class PerfTimer(object):
    """Measure execution time using :func;`time.perf_counter`.

    >>> with PerfTimer('do something'):
    >>>     do_something()
    do something: 2.49972483800957 seconds

    >>> with PerfTimer('do something', verbose=False) as t:
    >>>     do_something()
    >>> t.elapsed
    2.49972483800957
    """
    def __init__(self, name, verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self._end = time.perf_counter()
        self.elapsed = self._end - self._start
        if self.verbose:
            print(str(self))

    def __str__(self):
        return '{}: {} seconds'.format(self.name, self.elapsed)


def complex_plane(n, m, min, max):
    """Create n*m grid of complex numbers over ``[min, max)``."""
    assert min.real < max.real and min.imag < max.imag
    x = np.linspace(min.real, max.real, n, endpoint=False)
    y = np.linspace(min.imag * 1j, max.imag * 1j, m, endpoint=False)
    return x[:, None] + y[None, :]