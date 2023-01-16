#정수를 원하는 자릿수의 2진수로 변환하는 num2bin입니다.

def num2bin(num,bit):
    bin=format(num,'b').zfill(bit)
    return bin