import pickle
import numpy as np
from Multi_layer_NN import Multi_layer_NN
from Sigmoid import Sigmoid
from Multi_layer_NN import Multi_layer_NN
from Test_MNN import Test_MNN
import cv2
from im2double import im2double
from TestS2L import TestS2L
import matplotlib.pyplot as plt
import matplotlib

"""
4. 3번에서 넘어온 14bit segment LEDs코드를 입력 받아 
LED 이미지로 변환하는 코드입니다.
"""

#3번 데이터 불러오기
with open('E:\\Pycharm\\project01\\data_for_next\\xForSeg14.p','rb') as file: #(1x14)
    xForSeg14=pickle.load(file)
with open('E:\\Pycharm\\project01\\data_for_next\\xForSeg14TrainData.p','rb') as file: #(300x14)
    X=pickle.load(file)

X1=xForSeg14[0,0:7] #(1x7), 첫 번째 코드 추출
X2=xForSeg14[0,7:14] #(1x7), 두 번째 코드 추출

#Segment LED table
seg_LED=np.zeros((11,7),dtype=int)
seg_LED[0] = [0,0,0,0,0,0,1] #- (음수 부호)
seg_LED[1] = [1,1,1,1,1,1,0] #0
seg_LED[2] = [0,1,1,0,0,0,0] #1
seg_LED[3] = [1,1,0,1,1,0,1] #2
seg_LED[4] = [1,1,1,1,0,0,1] #3
seg_LED[5] = [0,1,1,0,0,1,1] #4
seg_LED[6] = [1,0,1,1,0,1,1] #5
seg_LED[7] = [1,0,1,1,1,1,1] #6
seg_LED[8] = [1,1,1,0,0,1,0] #7
seg_LED[9] = [1,1,1,1,1,1,1] #8
seg_LED[10] = [1,1,1,1,0,1,1] #9

#output data setting
D=np.zeros((11,784)) #(28x28) -> (1x784) 로 변환한 11개의 LED 이미지 데이터
for k in range(11):
    D[k, :] = np.reshape(im2double(cv2.imread('E:\\Pycharm\\project01\\data_led\\' + str(k) + '.png', cv2.IMREAD_GRAYSCALE)),(1, 784))

Df=np.zeros((2,300,784)) #최종 정답, 2개의 (300x784) 레이어, 각각이 LED 한 개(28x28)로 변환 됨
for k in range(300):
    if (X[k,0:7]==seg_LED[0,:]).all(): #첫 번째 자리 음수일 경우
        Df[0,k,:]=D[10,:] #첫 번째 자리 부호 그림
        if (X[k,7:14]==seg_LED[1,:]).all(): #두 번째 자리 0일 경우
            Df[1, k, :] = D[0, :] #0
        elif(X[k,7:14]==seg_LED[2,:]).all(): #두 번째 자리 1일 경우
            Df[1, k, :] = D[1, :]  # 1
        elif (X[k, 7:14] == seg_LED[3, :]).all():  # 두 번째 자리 2일 경우
            Df[1, k, :] = D[2, :]  # 2
        elif (X[k, 7:14] == seg_LED[4, :]).all():  # 두 번째 자리 3일 경우
            Df[1, k, :] = D[3, :]  # 3
        elif (X[k, 7:14] == seg_LED[5, :]).all():  # 두 번째 자리 4일 경우
            Df[1, k, :] = D[4, :]  # 4
        elif (X[k, 7:14] == seg_LED[6, :]).all():  # 두 번째 자리 5일 경우
            Df[1, k, :] = D[5, :]  # 5
        elif (X[k, 7:14] == seg_LED[7, :]).all():  # 두 번째 자리 6일 경우
            Df[1, k, :] = D[6, :]  # 6
        elif (X[k, 7:14] == seg_LED[8, :]).all():  # 두 번째 자리 7일 경우
            Df[1, k, :] = D[7, :]  # 7
        elif (X[k, 7:14] == seg_LED[9, :]).all():  # 두 번째 자리 8일 경우
            Df[1, k, :] = D[8, :]  # 8
        else: #두 번째 자리 9인 경우
            Df[1, k, :] = D[9, :]  # 9
    else: #양수일 경우
        if (X[k,0:7]==seg_LED[1,:]).all(): #첫 번째 자리 0일 경우
            Df[0, k, :] = D[0, :] #0
        elif(X[k,0:7]==seg_LED[2,:]).all(): #첫 번째 자리 1일 경우
            Df[0, k, :] = D[1, :]  # 1
        elif (X[k, 0:7] == seg_LED[3, :]).all():  # 첫 번째 자리 2일 경우
            Df[0, k, :] = D[2, :]  # 2
        elif (X[k, 0:7] == seg_LED[4, :]).all():  # 첫 번째 자리 3일 경우
            Df[0, k, :] = D[3, :]  # 3
        elif (X[k, 0:7] == seg_LED[5, :]).all():  # 첫 번째 자리 4일 경우
            Df[0, k, :] = D[4, :]  # 4
        elif (X[k, 0:7] == seg_LED[6, :]).all():  # 첫 번째 자리 5일 경우
            Df[0, k, :] = D[5, :]  # 5
        elif (X[k, 0:7] == seg_LED[7, :]).all():  # 첫 번째 자리 6일 경우
            Df[0, k, :] = D[6, :]  # 6
        elif (X[k, 0:7] == seg_LED[8, :]).all():  # 첫 번째 자리 7일 경우
            Df[0, k, :] = D[7, :]  # 7
        elif (X[k, 0:7] == seg_LED[9, :]).all():  # 첫 번째 자리 8일 경우
            Df[0, k, :] = D[8, :]  # 8
        else: #첫 번째 자리 9인 경우
            Df[0, k, :] = D[9, :]  # 9

        if (X[k,7:14]==seg_LED[1,:]).all(): #두 번째 자리 0일 경우
            Df[1, k, :] = D[0, :] #0
        elif(X[k,7:14]==seg_LED[2,:]).all(): #두 번째 자리 1일 경우
            Df[1, k, :] = D[1, :]  # 1
        elif (X[k, 7:14] == seg_LED[3, :]).all():  # 두 번째 자리 2일 경우
            Df[1, k, :] = D[2, :]  # 2
        elif (X[k, 7:14] == seg_LED[4, :]).all():  # 두 번째 자리 3일 경우
            Df[1, k, :] = D[3, :]  # 3
        elif (X[k, 7:14] == seg_LED[5, :]).all():  # 두 번째 자리 4일 경우
            Df[1, k, :] = D[4, :]  # 4
        elif (X[k, 7:14] == seg_LED[6, :]).all():  # 두 번째 자리 5일 경우
            Df[1, k, :] = D[5, :]  # 5
        elif (X[k, 7:14] == seg_LED[7, :]).all():  # 두 번째 자리 6일 경우
            Df[1, k, :] = D[6, :]  # 6
        elif (X[k, 7:14] == seg_LED[8, :]).all():  # 두 번째 자리 7일 경우
            Df[1, k, :] = D[7, :]  # 7
        elif (X[k, 7:14] == seg_LED[9, :]).all():  # 두 번째 자리 8일 경우
            Df[1, k, :] = D[8, :]  # 8
        else: #두 번째 자리 9인 경우
            Df[1, k, :] = D[9, :]  # 9

#Training
#Multi layer neural network for seg' LED code to LED images
W1 = 2*np.random.random((20,7))-1
W2 = 2*np.random.random((30,20))-1
W3 = 2*np.random.random((30,30))-1
W4 = 2*np.random.random((784,30))-1

#첫 번째 숫자
# for epoch in range(1000):
#      print(epoch)
#      W1,W2,W3,W4=Multi_layer_NN(W1,W2,W3,W4,X[:,0:7],Df[0,:,:])

# save weights
# with open('E:\\Pycharm\\project01\\data_for_next\\Seg2LED1.p','wb') as file:
#      pickle.dump(W1,file)
#      pickle.dump(W2, file)
#      pickle.dump(W3, file)
#      pickle.dump(W4, file)

# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\Seg2LED1.p', 'rb') as file:
    W1 = pickle.load(file)
    W2 = pickle.load(file)
    W3 = pickle.load(file)
    W4 = pickle.load(file)

num10 = np.zeros((28,28)) #십의 자리 LED 숫자
num10 = TestS2L(W1,W2,W3,W4,X1)

#두 번째 숫자
# for epoch in range(1000):
#      print(epoch)
#      W1,W2,W3,W4=Multi_layer_NN(W1,W2,W3,W4,X[:,7:14],Df[1,:,:])

# save weights
# with open('E:\\Pycharm\\project01\\data_for_next\\Seg2LED2.p','wb') as file:
#      pickle.dump(W1, file)
#      pickle.dump(W2, file)
#      pickle.dump(W3, file)
#      pickle.dump(W4, file)

# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\Seg2LED2.p', 'rb') as file:
    W1 = pickle.load(file)
    W2 = pickle.load(file)
    W3 = pickle.load(file)
    W4 = pickle.load(file)

num1 = np.zeros((28,28)) #일의 자리 LED 숫자
num1 = TestS2L(W1,W2,W3,W4,X2)

print("5. 14bit segment LEDs code -> LED images")
print("입력 :",xForSeg14)

plt.figure(figsize=(10,10))

plt.subplot(1,2,1)
plt.imshow(num10,cmap='gray')
plt.title("first LED image")

plt.subplot(1,2,2)
plt.imshow(num1,cmap='gray')
plt.title("second LED image")

plt.show()