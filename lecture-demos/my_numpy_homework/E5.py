import numpy as np
#把向量变为单位长度
def unit_vector(a):
    a=np.array(a,dtype=float)
    length=np.linalg.norm(a)
    if length==0:
        raise ValueError("a不能为零向量")

    result = a/length
    return result

a = np.array([-1.20, 0.40, 0.80])
result = unit_vector(a)
print("a的原向量为：", a)
print("a的单位向量为：", result)
print("a的单位向量长度为：", np.linalg.norm(result))
#np.allclose() 会判断两个数是否“足够接近”，误差可以忽略不计。
#它的作用是判断两个数是否相等，考虑到浮点数计算中可能存在的微小误差。
print("长度是否接近1=",np.allclose(np.linalg.norm(result), 1.0))