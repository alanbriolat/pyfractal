from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pyfractal.util import PerfTimer, complex_plane


def mandelbrot(n, m, min, max, itermax=100, threshold=2.0):
    """Calculate the Mandelbrot Set for the range ``[min, max)``, at a
    resolution of n*m.
    """
    with PerfTimer('setup'):
        # Get complex plane, and flatten it since axes don't matter during the
        # calculations - we can reshape at the end
        c = complex_plane(n, m, min, max).flatten()
        # Where we're going to store our iteration counts
        img = np.zeros_like(c, dtype=np.uint16)
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

    img.reshape((n, m))
    return img


if __name__ == '__main__':
    import pylab

    with PerfTimer('total'):
        min, max = -2 - 2j, 2 + 2j
        n, m = 2, 4
        w, h = 1024, 512
        ws = repeat(w)
        hs = repeat(h)
        mins = complex_plane(n, m, min, max).flatten()
        maxs = mins + complex((max.real - min.real) / n, (max.imag - min.imag) / m)
        iters = repeat(200)
        thresholds = repeat(2.0)

        with ProcessPoolExecutor(max_workers=2) as executor:
            imgs = list(executor.map(mandelbrot, ws, hs, mins, maxs, iters, thresholds))

        img = np.array(imgs).reshape((n, m, w, h)).transpose((0, 2, 1, 3)).reshape((n * w, m * h))

    pylab.imshow(img.T, origin='lower left')
    pylab.show()