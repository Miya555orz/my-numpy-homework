import numpy as np
A = np.array([[0.20, -0.50, 1.10, 0.40],
              [0.80, 0.30, -0.70, 0.60],
              [-0.40, 0.90, 0.10, -0.20]])
#维度
print("A.ndim=", A.ndim)
#形状(?)其实就是每个方向有多少个元素(3,4)就是三行四列
print("A.shape=", A.shape)
#元素总数
print("A.size=", A.size)

second_row=A[1]
#提取第二行的元素
print("second_row=", second_row)
#提取第二行的形状
print("second_row.shape=", second_row.shape)
#提取最后两列，区间用法是
#[所有行，2：4]，2:4表示从第2列到第4列(不包括第4列)
last_two_col=A[:, 2:4]
print("last_two_col=", last_two_col)
#提取最后两列的形状
last_two_col_shape=last_two_col.shape
print("last_two_col_shape=", last_two_col_shape)
#B为A的复制
B=A.copy()
#将B中小于0的元素设置为0
B[B<0]=0
print("B=", B)
