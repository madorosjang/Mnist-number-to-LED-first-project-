from Conv import Conv
from ReLU import ReLU
from MeanPool import MeanPool
import numpy as np
from Softmax import Softmax
from MaxPool import MaxPool
"""
미리 산출 된 가중치를 토대로 신경망을 활성화 시키는 TestConv입니다.
"""

def TestConv(W1,W5,Wo,X):
    y1=Conv(X,W1)
    y2=ReLU(y1)
    y3=MaxPool(y2,(2,2))
    y4=np.reshape(y3,(-1,1))
    v5=np.matmul(W5,y4)
    y5=ReLU(v5)
    v=np.matmul(Wo,y5)
    y=np.round(Softmax(v.T))

    return y