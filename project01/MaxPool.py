#def MaxPool

import numpy as np

def MaxPool(x,filter_size,stride=2):
    (xrow, xcol, numFilters) = x.shape  # (20x20x20) -> (xrow: 20, xcol: 20, numFilters: 20)
    yrow = int(((xrow - filter_size[0])/stride)+1)
    ycol = int(((xcol - filter_size[1])/stride)+1)
    y=np.zeros((yrow,ycol,numFilters)) # (10x10x20), pooling output

    for r in range(yrow):
        for c in range(ycol):
            for n in range(numFilters):
                slice_input=x[r * stride:r * stride + filter_size[0],c * stride:c * stride + filter_size[1],n]
                y[r,c,n] = np.max(slice_input)
    return y
