import numpy as np
#求出所有点对之间的距离矩阵
A = np.array([
    [0.20, -0.40],
    [0.80, 0.30],
    [-0.50, 0.70]
])

B = np.array([
    [0.10, 0.20],
    [-0.60, -0.20]
])

print("A.shape=", A.shape)
print("B.shape=", B.shape)

dif=A[:,None,:]-B[None,:,:]
print("dif=", dif)
print("dif.shape=", dif.shape)
#对平方和 开平方 得到距离
D=np.sqrt(np.sum(dif**2,axis=2))
print("D=", D)
print("D.shape=", D.shape)
#np.argmin(array,axis=1): 返回每一行最小值的索引
nearest_index=np.argmin(D,axis=1)
print("nearest_index=", nearest_index)

