from Multi_layer_NN import Multi_layer_NN
from Sigmoid import Sigmoid
import pickle
import numpy as np
from list2num import list2num
from Multi_layer_NN import Multi_layer_NN
from Test_MNN import Test_MNN


"""
3. 2번에서 넘어온 부호를 포함한 8bit 이진수를 입력 받아
14bit segment LED code로 변환하는 코드입니다.
"""

#2번 데이터 불러오기
with open('E:\\Pycharm\\project01\\data_for_next\\xForBi8.p','rb') as file: #(1x8)
    xForBi8=pickle.load(file)
with open('E:\\Pycharm\\project01\\data_for_next\\xForBi8TrainData.p','rb') as file: #(300x8)
    X=pickle.load(file)

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

D1=np.zeros((1,4),dtype=int)
D1[0,:]=[1,0,0,0] #음수(-) 부호
D=np.zeros((300,14),dtype=int) #최종 누적 seg'LED table

for k in range(300):
    if (X[k,0:4]==D1[0,:]).all(): #음수일 경우
        set10 = seg_LED[0] #십의 자리 -부호로 세팅
        num2 = X[k,4:8] #일의 자리 숫자 추출
        num2_10 = list2num(num2,4) #10진수 변환
        if num2_10==1:
            set1=seg_LED[2] #1
        elif num2_10==2:
            set1 = seg_LED[3] #2
        elif num2_10==3:
            set1 = seg_LED[4] #3
        elif num2_10==4:
            set1 = seg_LED[5] #4
        elif num2_10==5:
            set1 = seg_LED[6] #5
        elif num2_10==6:
            set1 = seg_LED[7] #6
        elif num2_10==7:
            set1 = seg_LED[8] #7
        elif num2_10==8:
            set1 = seg_LED[9] #8
        else:
            set1 = seg_LED[10] #9
        D[k,:]=np.hstack((set10,set1))

    else: #양수일 경우
        num=list2num(X[k,:],8) #8bit 이진수를 정수로 변환
        num1 = np.floor(num/10) #10의자리 숫자 판별, 소수점 이하 버림
        num2 = num%10 #1의 자리 숫자 판별
        if num1==0:
            set10=seg_LED[1] #0
        elif num1==1:
            set10 = seg_LED[2] #1
        elif num1==2:
            set10 = seg_LED[3] #2
        elif num1==3:
            set10 = seg_LED[4] #3
        elif num1==4:
            set10 = seg_LED[5] #4
        elif num1==5:
            set10 = seg_LED[6] #5
        elif num1==6:
            set10 = seg_LED[7] #6
        elif num1==7:
            set10 = seg_LED[8] #7
        elif num1==8:
            set10 = seg_LED[9] #8
        else:
            set10 = seg_LED[10] #9

        if num2==0:
            set1=seg_LED[1] #0
        elif num2==1:
            set1 = seg_LED[2] #1
        elif num2==2:
            set1 = seg_LED[3] #2
        elif num2==3:
            set1 = seg_LED[4] #3
        elif num2==4:
            set1 = seg_LED[5] #4
        elif num2==5:
            set1 = seg_LED[6] #5
        elif num2==6:
            set1 = seg_LED[7] #6
        elif num2==7:
            set1 = seg_LED[8] #7
        elif num2==8:
            set1 = seg_LED[9] #8
        else:
            set1 = seg_LED[10] #9
        D[k, :] = np.hstack((set10, set1))

#Training
#Multi layer neural network for 8bit to seg' LED code
W1 = 2*np.random.random((30,8))-1
W2 = 2*np.random.random((20,30))-1
W3 = 2*np.random.random((20,20))-1
W4 = 2*np.random.random((14,20))-1

# for epoch in range(1000):
#      print(epoch)
#      W1,W2,W3,W4=Multi_layer_NN(W1,W2,W3,W4,X,D)

#save weights
# with open('E:\\Pycharm\\project01\\data_for_next\\weights_8bit2Seg.p','wb') as file:
#      pickle.dump(W1,file)
#      pickle.dump(W2, file)
#      pickle.dump(W3, file)
#      pickle.dump(W4, file)

# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\weights_8bit2Seg.p', 'rb') as file:
          W1 = pickle.load(file)
          W2 = pickle.load(file)
          W3 = pickle.load(file)
          W4 = pickle.load(file)

y= Test_MNN(W1,W2,W3,W4,xForBi8) #(1x14)
print("3. 8bit binary code -> 14bit segment LEDs code")

if (xForBi8[0,0:4]==D1[0,:]).all(): #음수일 경우
    op='-'
    set1=list2num(xForBi8[0,4:8],4) #일의 자리 숫자 추출
    print("입력 (%s %d) -> "%(op,set1),xForBi8)
    print("\n출력 (%s %d) -> " %(op,set1), y)
else: #양수일 경우
    num = list2num(xForBi8[0,:], 8)  # 정수 추출
    set10 = np.floor(num/10) #십의 자리 숫자 추출
    set1 = num%10 #일의 자리 숫자 추출
    print("입력 (%d %d) -> " % (set10, set1), xForBi8)
    print("\n출력 (%d) -> " % (num), y)

xForSeg14=y #(1,14), 4번의 input
xForSeg14TrainData = D #(300,14), 4번의 training input data

#3번 데이터 저장
with open('E:\\Pycharm\\project01\\data_for_next\\xForSeg14.p','wb') as file:
     pickle.dump(xForSeg14,file)

with open('E:\\Pycharm\\project01\\data_for_next\\xForSeg14TrainData.p', 'wb') as file:
    pickle.dump(xForSeg14TrainData, file)




