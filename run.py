import time
import pylab
import mandelbrot

before = time.perf_counter()
img = mandelbrot.mandelbrot(1500, 1500, -2, 2, -2, 2, 200, 2.0)
after = time.perf_counter()
print("took {} seconds".format(after - before))

pylab.imshow(img.T, origin='lower left')
pylab.show()