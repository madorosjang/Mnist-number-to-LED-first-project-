# def Softmax
import numpy as np

def Softmax(x):
    x  = np.subtract(x, np.max(x))        # prevent overflow, 오버플로 방지
    ex = np.exp(x)
    
    return ex / np.sum(ex)


