import numpy as np
#计算两个点之间的距离
p = np.array([-0.50, 1.20, 2.00])
q = np.array([0.80, -0.40, 1.10])

#两个点的坐标差
dif=p-q
print("dif=", dif)

#欧式距离
dis=np.sqrt(np.sum(dif**2))
print("dis=", dis)

#norm验证
print("norm_dis=", np.linalg.norm(dif))

#曼哈顿距离
mdis=np.sum(np.abs(dif))
print("mdis=", mdis)