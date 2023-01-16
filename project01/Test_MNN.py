import numpy as np
from Sigmoid import Sigmoid
#앞서 산출된 가중치를 바탕으로 신경망을 활성화 시키는 Test_MNN입니다.

def Test_MNN(W1,W2,W3,W4,X):
    v1 = np.matmul(W1,X.T)
    y1 = Sigmoid(v1)

    v2 = np.matmul(W2,y1)
    y2 = Sigmoid(v2)

    v3 = np.matmul(W3,y2)
    y3 = Sigmoid(v3)

    v = np.matmul(W4,y3)
    y=np.round(Sigmoid(v.T))

    return y