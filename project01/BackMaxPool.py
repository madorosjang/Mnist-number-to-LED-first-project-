#def BackMaxPool, maxpooling의 backpropagation을 위한 table을 출력합니다.
import numpy as np

def BackMaxPool(x,filter_size,stride=2):
    (xrow, xcol, numFilters) = x.shape  # (20x20x20) -> (xrow: 20, xcol: 20, numFilters: 20)
    yrow = int(((xrow - filter_size[0]) / stride) + 1)
    ycol = int(((xcol - filter_size[1]) / stride) + 1)
    y=np.zeros((xrow,xcol,numFilters)) #(20x20x20), 원래 크기로 다시 복구

    for n in range(numFilters):
        for r in range(yrow):
            for c in range(ycol):
                slice_x = x[r * stride : r * stride + filter_size[0], c * stride : c * stride + filter_size[1],n]
                max_vec=np.max(slice_x) #(2x2) 중 max 값 추출
                #max값을 기준으로 filter 크기(2x2)안에서 max 위치 판별(max면 1, 아니면 0)
                for n1 in range(2):
                    for n2 in range(2):
                        if x[r * stride + n1 , c * stride + n2,n] >=max_vec:
                            y[r * stride + n1, c * stride + n2, n] = 1
                        else:
                            y[r * stride + n1, c * stride + n2, n] = 0

    return y