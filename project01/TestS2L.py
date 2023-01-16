import numpy as np
from Sigmoid import Sigmoid
def TestS2L(W1,W2,W3,W4,X):
    v1 = np.matmul(W1,X.T)
    y1 = Sigmoid(v1)

    v2 = np.matmul(W2,y1)
    y2 = Sigmoid(v2)

    v3 = np.matmul(W3,y2)
    y3 = Sigmoid(v3)

    v = np.matmul(W4,y3)
    y=np.reshape(Sigmoid(v.T),(28,28))

    return y