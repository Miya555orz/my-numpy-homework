import numpy as np
#沿不同axis汇总
A = np.array([
    [0.20, -0.50, 1.10],
    [0.80, 0.40, -0.30]
])
print("A.shape=", A.shape)
#axis=0表示沿着列方向汇总，axis=1表示沿着行方向汇总
#先把上下两行合并
col_sum=A.sum(axis=0)

print("col_sum=", col_sum)
print("col_sum.shape=", col_sum.shape)

#axis=1可以沿着行方向汇总
#把每行钟的三列合并
#mean()函数可以求平均值
row_mean=A.mean(axis=1)
print("row_mean=", row_mean)
print("row_mean.shape=", row_mean.shape)

max_value=A.max()
print("max_value=", max_value)