import numpy as np
from MaxPool import MaxPool
from BackMaxPool import BackMaxPool
x1=np.zeros((20,20,20))
x=np.random.randn(20,20)

for k in range(20):
    x1[:,:,k]=x*(k+1)

print(x1)
y=MaxPool(x1,(2,2))
print(y)

y1=BackMaxPool(x1,(2,2))
print(y1)


