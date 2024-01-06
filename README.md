# Mnist-number-to-LED-first-project-
Mnist 손글씨 인식 신경망 구축 (개인 프로젝트)
- 인공지능 과목 기말 프로젝트(by MATLAB) 파이썬으로 구현

### 진행 기간
- 22.12.26~23.01.08

### 개발 환경
- Python
- IDE : PyCharm

### 프로젝트 설명
- 숫자 2개와 연산자(더하기,빼기,곱하기) 1개를 선택
- 숫자는 10bit One-hot-encoding(이하 'OHE'), 연산자는 3bit OHE로 변환
- 숫자 OHE에 따른 정수는 각각 4bit 2진수로 변환
- 연산자 OHE와 결합 되어 11bit 계산코드(숫자+연산자+숫자)로 변환
- 연산 결과를 Segment LED코드로 변환
- Segment LED코드를 28x28픽셀 LED 이미지로 변환
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213192703-7de7e1e4-81b6-4c99-9e06-c715f872fe66.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179501-d5ffd5c1-d582-4770-b6cd-9097ac12b376.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179544-eb44b608-fbd8-4e8a-a3c2-cb8ca107931f.JPG"/>

### 프로젝트 실행 방법
- 코드 TestPic2OHE, TestOHE2bit_8, Test8bit2Seg, TestSeg2LED를 순서대로 실행
- 사용자에 따른 새로운 경로 설정 필요 
```
# load weights
with open('E:\\Pycharm\\project01\\data_for_next\\weights_Pic_Conv.p', 'rb') as file:
          W1=pickle.load(file)
          W5=pickle.load(file)
          Wo=pickle.load(file)
```

<img width="100%" src="https://user-images.githubusercontent.com/122807795/213181421-bf036641-3810-4f57-b500-0614b62c498d.gif"/>

### 인공지능 구현 방법
- Mnist database에서 가져온 t10k-images-idx3-ubyte의 1만개 이미지 파일과 t10k-labels-idx1-ubyte의 1만개 라벨을 이용해 미리 학습시킨 convolution neural network을 거쳐 나온 가중치로 숫자를 판별
- 그림판으로 직접 그린 50개의 연산자 이미지와 그에 따른 50개의 라벨을 이용해 미리 학습시킨 convolution neural network을 거쳐 나온 가중치로 연산자 판별
- convolution neural network의 활성함수는 ReLU를 사용
- output layer는 OHE판별을 위해 Softmax를 사용
- 가중치 계산은 숫자와 연산자 각각 100개, 10개 씩 평균을 내며 계산하는 미니배치 방식을 사용
- Pooling은 Meanpooling 과 Maxpooling을 둘 다 사용해 보았고 최종 코드는 Maxpooling을 사용
- 숫자와 연산자 판별 이후 나머지 과정들은 3개의 hidden layer를 가진 Multi neural network를 통해 학습 및 판별
- Multi neural network의 활성함수는 Sigmoid를 사용
- 학습과정에 필요한 training 데이터와 정답 데이터는 반복문과 함수를 통해 생성

### 코드 출처
- MATLAB Deep Learning(출판사 : Apress, 저자 : Phill Kim)
- convolution neural network 및 Multi neural network, Meanpooling 는 교수님의 코드 사용
- Maxpooling은 Open AI의 ChatGPT 검색 결과를 재구성해 작성
- Maxpooling의 Backpropagation 과정(BackMaxPool)은 인터넷 자료 참고하여 본인이 작성 

### 프로젝트 후 배운 점
- convolution neural network, Multi neural network, 신경망 설계 방법 등 인공지능 기초 지식 습득
- 파이썬의 문법, 라이브러리 설치 방법, 함수 생성 법 등 파이썬 기초 지식 습득

___
#### 참고자료 
- 미상의 작성자, ratsgo's blog, <https://ratsgo.github.io/deep%20learning/2017/04/05/CNNbackprop/>, 2017.04.05
- 미상의 작성자, MATLAB Answers, <https://kr.mathworks.com/matlabcentral/answers/409032-how-do-i-compute-the-maxpool-of-a-image-let-us-say-stride-of-2-2-on-a-mxn-matrix>, 2018.07.06
- 도움말센터 관리자, MathWorks, <https://kr.mathworks.com/help/matlab/ref/mat2cell.html>, 2022
- ChatGPT, OpenAI, <https://chat.openai.com/chat/beb03c74-f0d1-48c8-962e-f06cbec8883c>, 2023.01.19

