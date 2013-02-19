import time
import numpy as np


class PerfTimer(object):
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


def mandelbrot(n, m, xmin, xmax, ymin, ymax, itermax=100, threshold=2.0):
    """NumPy-based Mandelbrot set calculation, based on
    http://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    """
    with PerfTimer('setup'):
        # Create n*m grid of complex numbers covering the x-y range
        ix, iy = np.mgrid[0:n, 0:m]
        x = np.linspace(xmin, xmax, n)[ix]
        y = np.linspace(ymin, ymax, m)[iy]
        c = x + complex(0, 1) * y
        # Free up some memory
        del x, y, ix, iy

        # Flatten arrays, since dimensions don't matter for the calculations
        # (we can reshape for the final result)
        c.shape = n * m
        # Where we're going to store our iteration counts
        img = np.zeros(c.shape, dtype=np.uint16)
        # The points that still remain
        r = np.arange(n * m, dtype=int)

        # z0 = c
        z = np.copy(c)

    with PerfTimer('calculate'):
        for i in range(itermax):
            # Apply 'z = z*z + c' in-place
            np.multiply(z, z, z)
            np.add(z, c, z)
            # Find points that have escaped
            done = abs(z) > threshold
            # Save the iteration counts
            img[r[done]] = i + 1
            # Remove points that have escaped
            rem = -done
            r = r[rem]
            z = z[rem]
            c = c[rem]

    img.shape = (n, m)
    return img


if __name__ == '__main__':
    import pylab

    with PerfTimer('total'):
        img = mandelbrot(1500, 1500, -2, 2, -2, 2, 200, 2.0)

    pylab.imshow(img.T, origin='lower left')
    pylab.show()