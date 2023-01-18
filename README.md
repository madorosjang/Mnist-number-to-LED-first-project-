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
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213192703-7de7e1e4-81b6-4c99-9e06-c715f872fe66.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179501-d5ffd5c1-d582-4770-b6cd-9097ac12b376.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179544-eb44b608-fbd8-4e8a-a3c2-cb8ca107931f.JPG"/>

### 인공지능 구현 방법
- Mnist database에서 가져온 t10k-images-idx3-ubyte의 1만개 이미지 파일과 t10k-labels-idx1-ubyte의 1만개 라벨을 이용해 미리 학습시킨 convolution neural network을 거쳐 나온 가중치로 숫자를 판별했습니다. 연산자 또한 그림판으로 직접 그린 50개의 연산자 이미지와 그에 따른 50개의 라벨을 이용해 미리 학습시킨 convolution neural network을 거쳐 나온 가중치로 판별했습니다. 
- convolution neural network의 활성함수는 ReLU를 사용하였고 output layer는 OHE판별을 위해 Softmax를 사용했습니다. 가중치 계산은 숫자와 연산자 각각 100개, 10개 씩 평균을 내며 계산하는 미니배치 방식을 사용하였고 Pooling은 Meanpooling 과 Maxpooling을 둘 다 사용해 보았고 최종 코드는 Maxpooling을 사용했습니다. 이 때 Meanpooling이 Maxpooling보다 빠른 학습속도를 보였으나 숫자 판별 시 6을 0으로 인식하는 등 인식률이 떨어지는 모습을 보였고, Maxpooling은 학습속도는 느리나 모든 숫자 및 연산자를 인식하며 더 높은 인식률을 보였습니다. 
- 숫자와 연산자 판별 이후 나머지 과정들은 3개의 hidden layer를 가진 Multi neural network를 통해 학습 및 판별이 이뤄졌으며 이 때 활성함수는 Sigmoid를 사용하였고 학습과정에 필요한 training 데이터와 정답 데이터는 반복문과 함수 생성을 통해 직접 만들었습니다. 마지막 출력 LED 이미지 또한 그림판으로 직접 만들었습니다. 
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213195986-65a1e837-8092-4f55-88f5-c01bbc68a09c.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213195998-a3aa763b-93a7-4209-b7d0-2ee52fe4da3f.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213196007-95ea56e9-913a-4e2d-9bf3-7c3074747c0d.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213196015-3884d72e-14fb-43a3-8b98-4a4c5817715e.JPG"/>
