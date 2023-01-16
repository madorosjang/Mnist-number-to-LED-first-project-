import numpy as np
#문자열을 리스트로 변환해주는 str2list입니다.

def str2list(bin,bit):
    strList = np.zeros((1, bit), dtype=int)
    for k in range(bit):
        strList[:, k] = bin[k:k + 1]
    return strList