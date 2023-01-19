import random
import numpy as np
import pickle
import cv2
from im2double import im2double
from LoadMnistData import LoadMnistData
from MnistConv import MnistConv
from Conv import Conv
from TestConv import TestConv
import matplotlib.pyplot as plt
import matplotlib
from pytictoc import TicToc

"""
1. 손글씨 숫자 2개와 연산자 1개를 입력 받아 각각 
10bit, 3bit one hot encoding으로 변환하는 코드입니다.
"""

# Images, Labels = LoadMnistData('E:\\Pycharm\\project01\\data\\t10k-images-idx3-ubyte.gz', 'E:\\Pycharm\\project01\\data\\t10k-labels-idx1-ubyte.gz')
# Images = np.divide(Images, 255)
#
# X = Images[:,:,:] #train input
# D = Labels[:] #correct output

#save Images_and_Labels
# with open('Images_and_Labels.p','wb') as file:
#      pickle.dump(X, file)
#      pickle.dump(D, file)

# load Images_and_Labels
with open('Images_and_Labels.p', 'rb') as file:
          X=pickle.load(file)
          D=pickle.load(file)

#연산자 트레이닝 데이터
Df=np.zeros((50,784)) #연산자 누적
Dff=np.zeros((50,28,28)) #연산자 최종 정답
D_labels = np.zeros((50,1),dtype=int) #연산자 라벨 (뺄셈:0, 덧셈:1, 곱셈:2)

for k in range(50):
     Df[k,:] = np.reshape(im2double(cv2.imread('E:\\Pycharm\\project01\\data_for_op\\'+str(k+1)+'.png',cv2.IMREAD_GRAYSCALE)),(1,784))
     Dff[k,:,:]=np.reshape(Df[k,:].T,(28,28))

for k in range(50):
     if k<10or(k>=30and k<40): #곱셈의 경우
          D_labels[k,:]=2
     elif (k>=10and k<20)or(k>=40and k<50): #덧셈의 경우
          D_labels[k,:]=1
     else: #뺄셈의 경우
          D_labels[k,:]=0

#1. 숫자 Training
W1=1e-2*np.random.randn(9,9,20)
W5 = np.random.uniform(-1, 1, (100, 2000)) * np.sqrt(6) / np.sqrt(360 + 2000)
Wo = np.random.uniform(-1, 1, ( 10,  100)) * np.sqrt(6) / np.sqrt( 10 +  100)

t=TicToc() #시간 측정 위한 TicToc 클래스 생성
# t.tic()
# for epoch in range(5):
#      print(epoch+1)
#      W1,W5,Wo = MnistConv(W1,W5,Wo,X,D)
# end=t.tocvalue()
# print("손글씨 training 소요 시간(MaxPooling) : %.0f분 %.0f초"%(np.floor(end/60), end-(np.floor(end/60)*60)))
# #save weights
# with open('E:\\Pycharm\\project01\\data_for_next\\weights_Pic_Conv.p','wb') as file:
#      pickle.dump(W1,file)
#      pickle.dump(W5, file)
#      pickle.dump(Wo, file)

# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\weights_Pic_Conv.p', 'rb') as file:
          W1=pickle.load(file)
          W5=pickle.load(file)
          Wo=pickle.load(file)

#2. 연산자 Training
W1_op=1e-2*np.random.randn(9,9,20)
W5_op = np.random.uniform(-1, 1, (100, 2000)) * np.sqrt(6) / np.sqrt(360 + 2000)
Wo_op = np.random.uniform(-1, 1, ( 3,  100)) * np.sqrt(6) / np.sqrt( 10 +  100)

# t.tic()
# for epoch in range(100):
#     print(epoch+1)
#     W1_op, W5_op, Wo_op = MnistConv(W1_op, W5_op, Wo_op, Dff, D_labels)
# end=t.tocvalue()
# print("연산자 training 소요 시간(MaxPooling) : %.0f분 %.0f초"%(np.floor(end/60), end-(np.floor(end/60)*60)))

# #save weights
# with open('E:\\Pycharm\\project01\\data_for_next\\weights_Pic_Conv_op.p','wb') as file:
#      pickle.dump(W1_op,file)
#      pickle.dump(W5_op, file)
#      pickle.dump(Wo_op, file)

# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\weights_Pic_Conv_op.p', 'rb') as file:
          W1_op=pickle.load(file)
          W5_op=pickle.load(file)
          Wo_op=pickle.load(file)

#연산자 테이블
D2=np.zeros((3,3))
D2[0,:]=[0,0,1]  #곱셈
D2[1,:]=[0,1,0]  #덧셈
D2[2,:]=[1,0,0] #뺄셈

n1=random.randrange(0,10) #임의의 손글씨1 추출 위한 숫자
n2=random.randrange(0,10) #임의의 손글씨2 추출 위한 숫자
n3=random.randrange(0,3) #임의의 연산자 추출 위한 숫자

#input 숫자 선택
X_num=np.zeros((10,784)) #숫자 input image
X_numf=np.zeros((10,28,28)) #숫자 input image to (28x28)
for k in range(10):
     X_num[k, :] = np.reshape(im2double(cv2.imread('E:\\Pycharm\\project01\\data_for_num\\' + str(k) + '.png', cv2.IMREAD_GRAYSCALE)),(1, 784))
     X_numf[k, :, :] = np.reshape(X_num[k, :].T, (28, 28))

X1=X_numf[n1,:,:] #n1에 대응되는 손글자 input
X2=X_numf[n2,:,:] #n2에 대응되는 손글자 input

xForNum1=np.zeros((1,10)) #2번의 input으로 들어갈 onehot encoding num1
xForNum2=np.zeros((1,10)) #2번의 input으로 들어갈 onehot encoding num2

#nput 연산자 선택
X_op=np.zeros((3,784)) #연산자 input image
X_opf=np.zeros((3,28,28)) #연산자 input image to (28x28)
for k in range(3):
    X_op[k, :] = np.reshape(im2double(cv2.imread('E:\\Pycharm\\project01\\data_for_op_input\\' + str(k) + '.png', cv2.IMREAD_GRAYSCALE)),(1, 784))
    X_opf[k, :, :] = np.reshape(X_op[k, :].T, (28, 28))

X3=X_opf[n3,:,:] #n3에 대응되는 연산자 input
xForOp=np.zeros((1,3)) #2번의 input으로 들어갈 onehot encoding operator

#첫 번째 숫자 Output
xForNum1=TestConv(W1,W5,Wo,X1) #1x10

#두 번째 숫자 Output
xForNum2=TestConv(W1,W5,Wo,X2) #1x10

#연산자 Output
xForOp=TestConv(W1_op,W5_op,Wo_op,X3) #1x3

print("1. 숫자 및 연산자 손글씨 -> One Hot Encoding code")

plt.figure(figsize=(10,10))

plt.subplot(1,3,1)
plt.imshow(X1,cmap='gray')
plt.title("first input")

plt.subplot(1,3,2)
plt.imshow(X3,cmap='gray')
plt.title("operator")

plt.subplot(1,3,3)
plt.imshow(X2,cmap='gray')
plt.title("second input")


if n3 == 0: #곱셈일 경우
    op='x'
elif n3 == 1: #덧셈일 경우
    op='+'
else: #뺄셈일 경우
    op='-'

print("첫 번째 출력 (정답 : %d): "%(n1),xForNum1)
print("\n연산자 (%s) : "%(op),xForOp)
print("\n두 번째 출력 (정답 : %d): "%(n2),xForNum2)

#1번 데이터 저장
with open('E:\\Pycharm\\project01\\data_for_next\\xForNum1.p', 'wb') as file: #1x10
    pickle.dump(xForNum1, file)
with open('E:\\Pycharm\\project01\\data_for_next\\xForNum2.p', 'wb') as file: #1x10
    pickle.dump(xForNum2, file)
with open('E:\\Pycharm\\project01\\data_for_next\\xForOp.p', 'wb') as file: #1x3
    pickle.dump(xForOp, file)

plt.show()






