import numpy as np

a = np.array([1,2,3,4,5,6])
print(a.shape)
a = a.reshape(-1,3)
print(a.shape)