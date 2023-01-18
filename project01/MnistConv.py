import numpy as np
from Conv import Conv
from ReLU import ReLU
from MaxPool import MaxPool
from project01.BackMaxPool import BackMaxPool
from Softmax import Softmax
from scipy import signal
"""
28x28이미지를 Onehotencoding으로 변환해주는 MnistConv입니다. 10개 혹은 100개마다 평균을 내서 가중치를 산출하는 미니배치기법을 사용합니다.
"""

def MnistConv(W1, W5, Wo, X, D):
    # W1 = conv, W5 = classifi' hidden layer, Wo = output layer

    N = len(D)  # 10000 or 50

    alpha = 0.01  # Learning rate
    beta = 0.95  # Momentum coefficient

    # momentums for weights
    momentum1 = np.zeros_like(W1)  # (9x9x20)
    momentum5 = np.zeros_like(W5)  # (100x2000)
    momentumo = np.zeros_like(Wo)  # (10x100) or (3x100)

    if N<=50: #연산자 CNN일 경우
        bsize= 10 #10개마다 평균 내서 업데이트
    else: #Mnist CNN일 경우
        bsize = 100 #100개마다 평균 내서 업데이트

    blist = np.arange(0, N, bsize)  # 0 ~ 9999 step 100 (0, 100, 200, 300, ... , 9900)
                                    # 0 ~ 49 step 5(0,10,20, ... ,40)
    #One epoch loop
    for batch in range(len(blist)):  # 0 ~ 100 or 0~5
        dW1 = np.zeros_like(W1)  # 9x9x20
        dW5 = np.zeros_like(W5)  # 100x2000
        dWo = np.zeros_like(Wo)  # 10x100 or 3x100


        #Mini batch loop
        begin = blist[batch]  # 0, 100, 200, 300, ... , 9900 or 0,10,20, ... ,40

        for k in range(begin, begin + bsize):  # 0~99, 100~199, 200~299, ... , 9900~9999 or 0~9, 10~19, ... ,40~49
            # Forward pass = inference
            x = X[k, :, :]  # (28x28)
            y1 = Conv(x, W1)  # (28x28) * (9x9x20) -> (20x20x20)
            y2 = ReLU(y1)  # (20x20x20)
            y3 = MaxPool(y2,(2,2))  # (10x10x20)
            y4 = np.reshape(y3, (-1, 1))  # (2000x1)
            v5 = np.matmul(W5, y4)  # (100x2000) X (2000x1) -> (100x1)
            y5 = ReLU(v5)  # (100x1)
            v = np.matmul(Wo, y5)  # (10x100) X (100x1) -> (10x1) or (3x100) X (100x1) -> (3x1)
            y = Softmax(v)  # (10x1) or (3x1)

            # one-hot encoding
            if N<=50: #연산자 CNN일 경우
                d = np.zeros((3,1))
            else: #Mnist CNN일 경우
                d = np.zeros((10, 1))

            d[D[k][0]][0] = 1  #for one-hot encoding

            # Backpropagation
            e = d - y  # (10x1) or (3x1)
            delta = e  # (10x1) or (3x1), Cross Entropy

            e5 = np.matmul(Wo.T, delta)  # Hidden(ReLU) (100x10) X (10x1) -> (100x1) or (100x3) X (3x1) -> (100x1)
            delta5 = (y5 > 0) * e5  # (100x1)

            e4 = np.matmul(W5.T, delta5)  # Pooling layer (2000x100) X (100x1) -> (2000x1)

            e3 = np.reshape(e4, y3.shape)  # (2000x1) -> (10x10x20)

            e2 = np.zeros_like(y2)  # pooling (20x20x20)
            W3 = BackMaxPool(y2,(2,2))  # (20x20x20), max pool back propa' 위한 layer 생성. w3은 y2레이어의 각 pooling layer 별로 max값은 1, 나머지는 0을 저장.
            for c in range(20):  # 0 ~ 19
                # kron((10x10x20), (2x2)) -> (20x20x20)
                e2[:, :, c] = np.kron(e3[:, :, c], np.ones((2, 2))) * W3[:, :, c]

            delta2 = (y2 > 0) * e2  # (20x20x20), ReLU layer

            delta1_x = np.zeros_like(W1)  # (9x9x20), convolutional layer
            for c in range(20):  # 0 ~ 19
                # (28x28) * (20x20) -> (9x9)
                delta1_x[:, :, c] = signal.convolve2d(x[:, :], np.rot90(delta2[:, :, c], 2), 'valid')

            # Accumulation
            dW1 = dW1 + delta1_x  # (9x9x20)
            dW5 = dW5 + np.matmul(delta5, y4.T)  # (100x1) X (1x2000) -> (100x2000)
            dWo = dWo + np.matmul(delta, y5.T)  # (10x1) X (1x100) -> (10x100) or (3x1) X (1x100) -> (3x100)
        # Average
        dW1 = dW1 / bsize
        dW5 = dW5 / bsize
        dWo = dWo / bsize
        # Momentum
        momentum1 = alpha * dW1 + beta * momentum1
        W1 = W1 + momentum1

        momentum5 = alpha * dW5 + beta * momentum5
        W5 = W5 + momentum5

        momentumo = alpha * dWo + beta * momentumo
        Wo = Wo + momentumo

    return W1, W5, Wo