import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from LoadMnistData import LoadMnistData

images, labels =LoadMnistData('E:\\spyder\\project01\\data\\t10k-images-idx3-ubyte.gz','E:\\spyder\\project01\\data\\t10k-labels-idx1-ubyte.gz')
images=np.divide(images,255);
X=images[0:10000,:,:]
D=labels[0:10000]

img=np.reshape(X[0,:,:],(28,28))
plt.imshow(img, cmap=cm.binary)
plt.show()
lbl=D[0]
print(lbl)