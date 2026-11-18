import numpy as np
a=np.array([0.80, -0.60, 0.70])
b=np.array([-0.20, 0.90, 0.40])

#逐个元素的乘法
element_product=a*b
print("element_product=", element_product)
print("element_product.shape=", element_product.shape)

#点积
dot_product=np.dot(a,b) 
print("dot_product=", dot_product)
print("dot_product.shape=", dot_product.shape)

#平方求合开平方 求出向量长度
length_a=np.sqrt(np.sum(a**2))
print("length_a=", length_a)
length_b=np.sqrt(np.sum(b**2))
print("length_b=", length_b)

#norm验证
# (就是用 NumPy 自带的标准函数，检查手动计算的向量长度是否正确)
print("norm_a=", np.linalg.norm(a))
print("norm_b=", np.linalg.norm(b))