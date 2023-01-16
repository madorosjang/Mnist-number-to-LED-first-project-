from Sigmoid import Sigmoid
import numpy as np
#Sigmoid를 활성함수로 하는 Multi_layer_NN입니다.

def Multi_layer_NN(W1,W2,W3,W4,X,D):
    alpha=0.01
    N=300
    X_col=(np.shape(X))[1] #X의 열 개수 추출
    D_col=(np.shape(D))[1] #D의 열 개수 추출
    for k in range(N):
        x = np.reshape(X[k, :], (X_col, 1))
        d = np.reshape(D[k, :], (D_col, 1))


        #Forwarding start
        #first hidden layer
        v1=np.matmul(W1,x)
        y1=Sigmoid(v1)

        #second hidden layer
        v2=np.matmul(W2,y1)
        y2=Sigmoid(v2)

        #third hidden layer
        v3=np.matmul(W3,y2)
        y3=Sigmoid(v3)

        #output layer
        v=np.matmul(W4,y3)
        y=Sigmoid(v)
        #Forwarding end

        #Clac' error
        e=d-y

        #output hidden layer
        delta=e #Cross entropy

        #Backkproragation
        #third hidden layer
        e3=np.matmul(W4.T,delta)
        delta3=y3*(1-y3)*e3

        #second hidden layer
        e2=np.matmul(W3.T,delta3)
        delta2=y2*(1-y2)*e2

        #first hidden layer
        e1=np.matmul(W2.T,delta2)
        delta1=y1*(1-y1)*e1

        #calc' weight
        dW1 = alpha * delta1 * x.T
        dW2 = alpha * delta2 * y1.T
        dW3 = alpha * delta3 * y2.T
        dW4 = alpha * delta * y3.T

        W1 = W1 + dW1
        W2 = W2 + dW2
        W3 = W3 + dW3
        W4 = W4 + dW4

    return W1,W2,W3,W4
