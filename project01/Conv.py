# 컨볼루션을 수행하는 def Conv입니다.
import numpy as np
from scipy import signal

def Conv(x,W):
    (wrow, wcol, numFilters) = W.shape
    (xrow, xcol)=x.shape
    
    #Convolution result size
    yrow = xrow - wrow +1
    ycol = xcol - wcol +1
    
    y=np.zeros((yrow,ycol,numFilters))
    
    for k in range(numFilters):
        filter = W[:,:,k]
        filter = np.rot90(np.squeeze(filter),2) #(9x9x1)->(9x9) 180 degree rotation,np.rot90(input array,회전횟수)
        y[:,:,k] = signal.convolve2d(x,filter,'valid')

    return y

        


