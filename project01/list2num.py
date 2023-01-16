#list(2진수)를 정수로 바꿔주는 list2num입니다.

def list2num(list,bit):
    sum=0
    for k in range(bit):
        sum = sum + (int(list[k]) * (2 ** (bit - (k + 1))))
    return sum