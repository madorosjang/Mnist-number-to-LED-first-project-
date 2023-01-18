# Mnist-number-to-LED-first-project-
본 프로젝트는 대학교 학부 재학 중 수강했던 인공지능 강의의 기말고사 프로젝트를 파이썬으로 구현한 것입니다. (개인 프로젝트)

### 진행 기간
- 본 프로젝트(by MATLAB) : 22.11.15 ~ 22.12.06
- 파이썬 구현 : 22.12.26~23.01.08

### 개발 환경
- Python
- IDE : PyCharm

### 프로젝트 설명
- 28x28픽셀의 그림판으로 그린 숫자 2개와 연산자(더하기,빼기,곱하기) 1개를 선택한 뒤 이를 숫자는 10bit Onehotencoding(이하 'OHE'), 연산자 3bit OHE로 변환합니다. 
- 숫자 OHE에 따른 정수는 각각 4bit 2진수로 변환되고 이것이 연산자 OHE와 결합 되어 11bit 계산코드(숫자+연산자+숫자) 로 변환됩니다. 
- 계산코드의 연산 결과가 부호(음수 부호, 1 0 0 0)를 포함한 8bit 2진수로 변환되고 이를 다시 그에 해당하는 Segment LED코드로 변환합니다.
- 마지막으로 Segment LED코드를 두개의 28x28픽셀 LED 이미지로 변환합니다. 
- 최대한 주석을 많이 달아줌으로써 이해를 돕기 위해 노력했습니다. 
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213192703-7de7e1e4-81b6-4c99-9e06-c715f872fe66.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179501-d5ffd5c1-d582-4770-b6cd-9097ac12b376.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179544-eb44b608-fbd8-4e8a-a3c2-cb8ca107931f.JPG"/>

### 프로젝트 실행 방법
- 코드 TestPic2OHE, TestOHE2bit_8, Test8bit2Seg, TestSeg2LED를 순서대로 실행하면 알아서 숫자와 연산자를 선택하고 그에 따른 최종결과를 LED 이미지로 보여줍니다. 
- 이 때 환경이 변함에 따라 번거롭지만 새로운 경로를 설정해주어야 올바르게 실행됩니다. 
```
# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\weights_Pic_Conv.p', 'rb') as file:
          W1=pickle.load(file)
          W5=pickle.load(file)
          Wo=pickle.load(file)
```

<img width="100%" src="https://user-images.githubusercontent.com/122807795/213181421-bf036641-3810-4f57-b500-0614b62c498d.gif"/>

### 인공지능 구현 방법
- Mnist database에서 가져온 t10k-images-idx3-ubyte의 1만개 이미지 파일과 t10k-labels-idx1-ubyte의 1만개 라벨을 이용해 미리 학습시킨 convolution neural network을 거쳐 나온 가중치로 숫자를 판별했습니다. 연산자 또한 그림판으로 직접 그린 50개의 연산자 이미지와 그에 따른 50개의 라벨을 이용해 미리 학습시킨 convolution neural network을 거쳐 나온 가중치로 판별했습니다. 
- convolution neural network의 활성함수는 ReLU를 사용하였고 output layer는 OHE판별을 위해 Softmax를 사용했습니다. 가중치 계산은 숫자와 연산자 각각 100개, 10개 씩 평균을 내며 계산하는 미니배치 방식을 사용하였고 Pooling은 Meanpooling 과 Maxpooling을 둘 다 사용해 보았고 최종 코드는 Maxpooling을 사용했습니다. 이 때 Meanpooling이 Maxpooling보다 빠른 학습속도를 보였으나 숫자 판별 시 6을 0으로 인식하는 등 인식률이 떨어지는 모습을 보였고, Maxpooling은 학습속도는 느리나 모든 숫자 및 연산자를 인식하며 더 높은 인식률을 보였습니다. 
- 숫자와 연산자 판별 이후 나머지 과정들은 3개의 hidden layer를 가진 Multi neural network를 통해 학습 및 판별이 이뤄졌으며 이 때 활성함수는 Sigmoid를 사용하였고 학습과정에 필요한 training 데이터와 정답 데이터는 반복문과 함수 생성을 통해 직접 만들었습니다. 마지막 출력 LED 이미지 또한 그림판으로 직접 만들었습니다. 
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213195986-65a1e837-8092-4f55-88f5-c01bbc68a09c.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213195998-a3aa763b-93a7-4209-b7d0-2ee52fe4da3f.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213196007-95ea56e9-913a-4e2d-9bf3-7c3074747c0d.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213196015-3884d72e-14fb-43a3-8b98-4a4c5817715e.JPG"/>

### 코드 출처
- Sigmoid등 활성 함수는 "MATLAB Deep Learning(출판사 : Apress, 저자 : Phill Kim)"을 참고하였고, convolution neural network 및 Multi neural network는 교수님의 코드를 사용하였습니다. Meanpooling 또한 교수님의 코드를 가져왔으며 Maxpooling은 Open AI의 ChatGPT 검색 결과를 재구성해 작성하였으며 Maxpooling의 Backpropagation 과정(BackMaxPool)은 인터넷 검색을 통해 이론을 접한 뒤 코드로 구성하였습니다. 

### 프로젝트 후 배운 점
- 본 프로젝트를 통해 convolution neural network, Multi neural network가 어떻게 구성되고 원하는 결과를 얻기 위해선 신경망을 어떻게 설계해야하는 지 등 인공지능의 기초를 익혔습니다.
- 본래 MATLAB으로 작성 된 프로젝트를 파이썬으로 구현하는 과정에서 서로 다른 문법을 고쳐나가며 파이썬의 문법을 익힐 수 있었고 또한 다양한 파이썬 라이브러리 및 함수들을 사용하면서 파이썬의 기초를 익혔습니다.
- 학습데이터를 생성하는 과정에서 수 많은 디버깅을 통해 코딩의 기초를 익혔습니다.

___
#### 참고자료 
- 미상의 작성자, ratsgo's blog, <https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/>, 2017.04.05
- 미상의 작성자, MATLAB Answers, <https://kr.mathworks.com/matlabcentral/answers/409032-how-do-i-compute-the-maxpool-of-a-image-let-us-say-stride-of-2-2-on-a-mxn-matrix>, 2018.07.06
- 도움말센터 관리자, MathWorks, <https://kr.mathworks.com/help/matlab/ref/mat2cell.html>, 2022

