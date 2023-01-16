# def MeanPool
import numpy as np
from scipy import signal


def MeanPool(x):
    (xrow, xcol, numFilters) = x.shape  # (20x20x20) -> (xrow: 20, xcol: 20, numFilters: 20)
    y = np.zeros((int(xrow / 2), int(xcol / 2), numFilters))  # y: (10x10x20)

    for k in range(numFilters):  # 0 ~ 19
        filter = np.ones((2, 2)) / (2 * 2)
        # filter
        #     1  [1 1]
        #     -  [   ]
        #     4  [1 1]
        image = signal.convolve2d(x[:, :, k], filter, 'valid')  # x: (20x20), filter: (2x2), image: (19x19)

        y[:, :, k] = image[::2, ::2]  # image (0, 2, 4, 6, 8, 10, 12, 14, 16, 18) -> (10x10)
        # y: (10x10x20)

    return y

        


