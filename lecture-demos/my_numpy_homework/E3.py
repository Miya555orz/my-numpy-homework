import numpy as np
#计算两个向量的夹角
def angle_deg(a,b):
    a=np.array(a,dtype=float)
    b=np.array(b,dtype=float)
    #检查是不是都是一维数组
    if a.ndim!=1 or b.ndim!=1:
        raise ValueError("a,b必须为一维向量")

    if a.shape[0]!=b.shape[0]:
        raise ValueError("a,b必须为同维度向量")
    #计算向量长度
    length_a=np.linalg.norm(a)
    length_b=np.linalg.norm(b)
    #零向量没有方向，无法计算夹角
    if length_a==0 or length_b==0:
        raise ValueError("a,b不能为零向量")
    #计算cos theta
    cosine=(a@b)/(length_a*length_b)
    #arccos是弧度，clip是为了防止浮点数误差导致cosine>1或cosine<-1
    angle_rad=np.arccos(np.clip(cosine, -1.0, 1.0))
    #degrees是角度
    return np.degrees(angle_rad)

a = np.array([1.00, -1.00])
b = np.array([1.00, 0.00])

print("a与b的夹角为：", angle_deg(a,b), "度")
