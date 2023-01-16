import pickle
import numpy as np
from num2bin import num2bin
from str2list import str2list
from list2str import list2str
from list2num import list2num
from Multi_layer_NN import Multi_layer_NN
from Test_MNN import Test_MNN
import math

"""
2. 1에서 넘어온 두개의 10bit onehot encoding을 입력 받아 
각각 4자리 2진수로 변환 뒤 연산자와 합해 11bit code로 만들어준 뒤 
이에 대한 연산결과를 부호를 포함한 8bit 이진수 코드로 변환합니다.
"""
#1번 데이터 불러오기
with open('E:\\Pycharm\\project01\\data_for_next\\xForNum1.p', 'rb') as file: #1x10
    xForNum1= pickle.load(file)
with open('E:\\Pycharm\\project01\\data_for_next\\xForNum2.p', 'rb') as file: #1x10
    xForNum2=pickle.load(file)
with open('E:\\Pycharm\\project01\\data_for_next\\xForOp.p', 'rb') as file: #1x3
    xForOp=pickle.load(file)

for k in range (10):
    if(xForNum1[0][k]==1):
        num1=k #OHE to number

for k in range (10):
    if(xForNum2[0][k]==1):
        num2=k #OHE to number

bin1=num2bin(num1,4) #10진수 -> 4자리 2진수
bin2=num2bin(num2,4) #10진수 -> 4자리 2진수
binList1=str2list(bin1,4) #2진수 문자열 -> 1행 4열 리스트
binList2=str2list(bin2,4) #2진수 문자열 -> 1행 4열 리스트

assemble_code=np.hstack((binList1,xForOp,binList2)) #11bit code

#1. 11bit input data (300x11)
#연산자 테이블
X1=[[0,0,1]] #곱셈
X2=[[0,1,0]] #덧셈
X3=[[1,0,0]] #뺄셈

calc_all=np.zeros((300,11),dtype=int) #전체 경우의 수(곱셈, 덧셈, 뺄셈 순서)
mul=np.zeros((10,11),dtype=int) #곱셈의 경우의 수
mulAll=np.zeros((100,11),dtype=int) #곱셈의 경우의 수 전체 누적
sum=np.zeros((10,11)) #덧셈의 경우의 수
sumAll=np.zeros((100,11),dtype=int) #덧셈의 경우의 수 전체 누적
_min=np.zeros((10,11),dtype=int) #뺄셈의 경우의 수
minAll=np.zeros((100,11),dtype=int) #뺄셈의 경우의 수 전체 누적

for k in range(3):
    if k==0: #곱셈의 경우
        for k2 in range(10):
            for k3 in range(10):
                bin_k2=num2bin(k2,4)
                bin_k3=num2bin(k3,4)
                binList_k2= str2list(bin_k2,4)
                binList_k3 = str2list(bin_k3,4)
                mul[k3,:]=np.hstack((binList_k2,X1,binList_k3)) #10x11 곱셈 행렬
                if k2 == 0:
                    mulAll[0:10, :] = mul
                elif k2 == 1:
                    mulAll[10:20, :] = mul
                elif k2 == 2:
                    mulAll[20:30, :] = mul
                elif k2 == 3:
                    mulAll[30:40, :] = mul
                elif k2 == 4:
                    mulAll[40:50, :] = mul
                elif k2 == 5:
                    mulAll[50:60, :] = mul
                elif k2 == 6:
                    mulAll[60:70, :] = mul
                elif k2 == 7:
                    mulAll[70:80, :] = mul
                elif k2 == 8:
                    mulAll[80:90, :] = mul
                else:
                    mulAll[90:100, :] = mul

    if k==1: #덧셈의 경우
        for k2 in range(10):
            for k3 in range(10):
                bin_k2=num2bin(k2,4)
                bin_k3=num2bin(k3,4)
                binList_k2= str2list(bin_k2,4)
                binList_k3 = str2list(bin_k3,4)
                sum[k3,:]=np.hstack((binList_k2,X2,binList_k3)) #10x11 덧셈 행렬
                if k2 == 0:
                    sumAll[0:10, :] = sum
                elif k2 == 1:
                    sumAll[10:20, :] = sum
                elif k2 == 2:
                    sumAll[20:30, :] = sum
                elif k2 == 3:
                    sumAll[30:40, :] = sum
                elif k2 == 4:
                    sumAll[40:50, :] = sum
                elif k2 == 5:
                    sumAll[50:60, :] = sum
                elif k2 == 6:
                    sumAll[60:70, :] = sum
                elif k2 == 7:
                    sumAll[70:80, :] = sum
                elif k2 == 8:
                    sumAll[80:90, :] = sum
                else:
                    sumAll[90:100, :] = sum

    if k==2: #뺄셈의 경우
        for k2 in range(10):
            for k3 in range(10):
                bin_k2=num2bin(k2,4)
                bin_k3=num2bin(k3,4)
                binList_k2= str2list(bin_k2,4)
                binList_k3 = str2list(bin_k3,4)
                _min[k3,:]=np.hstack((binList_k2,X3,binList_k3)) #10x11 뺄셈 행렬
                if k2 == 0:
                    minAll[0:10, :] = _min
                elif k2 == 1:
                    minAll[10:20, :] = _min
                elif k2 == 2:
                    minAll[20:30, :] = _min
                elif k2 == 3:
                    minAll[30:40, :] = _min
                elif k2 == 4:
                    minAll[40:50, :] = _min
                elif k2 == 5:
                    minAll[50:60, :] = _min
                elif k2 == 6:
                    minAll[60:70, :] = _min
                elif k2 == 7:
                    minAll[70:80, :] = _min
                elif k2 == 8:
                    minAll[80:90, :] = _min
                else:
                    minAll[90:100, :] = _min

calc_all=np.vstack((mulAll,sumAll,minAll)) #(300x11) 총 연산 결과

#2. 8bit output data (300x8), 부호를 포함한 8bit 이진수 코드
D=np.zeros((300,8),dtype=int) #(300x8), 최종 누적 계산 결과(부호를 포함한 8bit 이진수)
D1=np.zeros((300,1),dtype=int) #(300x1), 최종 누적 계산 결과(10진수)
D2=[[1,0,0,0]] #음수(-) 부호

for k in range(3):
    for k2 in range(10):
        for k3 in range(10):
            if k==0:
                D1[(k*100)+(k2*10)+k3]=k2*k3
            elif k==1:
                D1[(k*100)+(k2*10)+k3]=k2+k3
            else:
                D1[(k * 100) + (k2 * 10) + k3] = k2 - k3

for k in range(300):
    if int(D1[k])<0:#음수인 경우
        bin = num2bin(int(D1[k])*-1, 4)
        set1=str2list(bin,4)
        D[k] = np.hstack((D2, set1))
    else:
        bin = num2bin(int(D1[k]), 8)
        bin2=str2list(bin,8)
        D[k]=bin2

#Training (11x1) in (8x1) out
#Multi layer neural network for 11bit to binary 8bit
W1 = 2*np.random.random((30,11))-1
W2 = 2*np.random.random((30,30))-1
W3 = 2*np.random.random((40,30))-1
W4 = 2*np.random.random((8,40))-1

# for epoch in range(1000):
#     print(epoch)
#     W1,W2,W3,W4=Multi_layer_NN(W1,W2,W3,W4,calc_all,D)

#save weights
# with open('E:\\Pycharm\\project01\\data_for_next\\weights_OHE2bit_8.p','wb') as file:
#      pickle.dump(W1,file)
#      pickle.dump(W2, file)
#      pickle.dump(W3, file)
#      pickle.dump(W4, file)

# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\weights_OHE2bit_8.p', 'rb') as file:
          W1 = pickle.load(file)
          W2 = pickle.load(file)
          W3 = pickle.load(file)
          W4 = pickle.load(file)

print("2. 11bit calculation code -> 8bit binary code")
y = Test_MNN(W1,W2,W3,W4,assemble_code) #(1x8) 출력

n=0
for k in range(3):
    if xForOp[0][k]==1:
        n=k

if n==2: #곱셈의 경우
    result_f = num1 * num2
    operator='x'
elif n==1: #덧셈의 경우
    result_f = num1 + num2
    operator='+'
else: #뺄셈의 경우
    result_f = num1 - num2
    operator='-'

print("\n입력 (%d %s %d) -> "%(num1,operator,num2), assemble_code)

if result_f<0: #음수의 경우
    result_f=result_f * (-1) #양수화
    print("\n출력 (%s %d) -> "%(operator,result_f),y)
else: #양수의 경우
    print("\n출력 (%d %d) -> "%(math.floor(result_f/10),(result_f%10)),y)

xForBi8=y #(1,8), 3번의 input
xForBi8TrainData = D #(300,8), 3번의 training input data

#2번 데이터 저장
with open('E:\\Pycharm\\project01\\data_for_next\\xForBi8.p','wb') as file:
     pickle.dump(xForBi8,file)

with open('E:\\Pycharm\\project01\\data_for_next\\xForBi8TrainData.p', 'wb') as file:
    pickle.dump(xForBi8TrainData, file)

