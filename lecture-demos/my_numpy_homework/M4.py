import numpy as np
#小型神经网络前向计算
X = np.array([
    [0.80, -0.60, 0.70],
    [-0.40, -0.50, 0.90]
])

W1 = np.array([
    [-0.10, 0.20, -0.80],
    [0.30, -0.50, 0.10]
])

b1 = np.array([-0.15, 0.05])

W2 = np.array([
    [0.20, -0.60],
    [-0.80, 0.52]
])

b2 = np.array([0.10, -0.20])

#T是矩阵的转置
Z=X@W1.T+b1

#ReLU激活函数
H=np.maximum(Z,0)

Y=H@W2.T+b2

print("Z=", Z)
print("Z.shape=", Z.shape)

print("H=", H)
print("H.shape=", H.shape)

print("Y=", Y)
print("Y.shape=", Y.shape)