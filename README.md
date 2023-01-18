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
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179501-d5ffd5c1-d582-4770-b6cd-9097ac12b376.JPG"/>
<img width="50%" src="https://user-images.githubusercontent.com/122807795/213179544-eb44b608-fbd8-4e8a-a3c2-cb8ca107931f.JPG"/>
