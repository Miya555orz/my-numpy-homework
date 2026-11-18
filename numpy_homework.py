"""NumPy数值计算练习：8道基础题和4道中等题。"""
import numpy as np


print("\n===== NP-E1 数组、形状与切片 =====")
A=np.array([[0.20,-0.50,1.10,0.40],
            [0.80,0.30,-0.70,0.60],
            [-0.40,0.90,0.10,-0.20]])
print("A.ndim=",A.ndim)
print("A.shape=",A.shape)
print("A.size=",A.size)
second_row=A[1]
last_two_col=A[:,2:4]
print("second_row=",second_row,"shape=",second_row.shape)
print("last_two_col=",last_two_col,"shape=",last_two_col.shape)
B=A.copy()
B[B<0]=0
print("B=",B)


print("\n===== NP-E2 逐元素乘法、点积与长度 =====")
a=np.array([0.80,-0.60,0.70])
b=np.array([-0.20,0.90,0.40])
element_product=a*b
dot_product=np.dot(a,b)
length_a=np.sqrt(np.sum(a**2))
length_b=np.sqrt(np.sum(b**2))
print("element_product=",element_product,"shape=",element_product.shape)
print("dot_product=",dot_product,"shape=",dot_product.shape)
print("length_a=",length_a,"norm_a=",np.linalg.norm(a))
print("length_b=",length_b,"norm_b=",np.linalg.norm(b))


print("\n===== NP-E3 两个向量的夹角 =====")
def angle_deg(a,b):
    a=np.array(a,dtype=float)
    b=np.array(b,dtype=float)
    if a.ndim!=1 or b.ndim!=1:
        raise ValueError("a,b必须为一维向量")
    if a.shape[0]!=b.shape[0]:
        raise ValueError("a,b必须为同维度向量")
    length_a=np.linalg.norm(a)
    length_b=np.linalg.norm(b)
    if length_a==0 or length_b==0:
        raise ValueError("a,b不能为零向量")
    cosine=(a@b)/(length_a*length_b)
    angle_rad=np.arccos(np.clip(cosine,-1.0,1.0))
    return np.degrees(angle_rad)

a=np.array([1.00,-1.00])
b=np.array([1.00,0.00])
print("a与b的夹角为：",angle_deg(a,b),"度")


print("\n===== NP-E4 两个点的距离 =====")
p=np.array([-0.50,1.20,2.00])
q=np.array([0.80,-0.40,1.10])
dif=p-q
dis=np.sqrt(np.sum(dif**2))
mdis=np.sum(np.abs(dif))
print("dif=",dif)
print("dis=",dis,"norm_dis=",np.linalg.norm(dif))
print("mdis=",mdis)


print("\n===== NP-E5 把向量变为单位长度 =====")
def unit_vector(a):
    a=np.array(a,dtype=float)
    length=np.linalg.norm(a)
    if length==0:
        raise ValueError("a不能为零向量")
    return a/length

a=np.array([-1.20,0.40,0.80])
result=unit_vector(a)
print("a的原向量为：",a)
print("a的单位向量为：",result)
print("a的单位向量长度为：",np.linalg.norm(result))
print("长度是否接近1=",np.allclose(np.linalg.norm(result),1.0))


print("\n===== NP-E6 沿不同axis汇总 =====")
A=np.array([[0.20,-0.50,1.10],
            [0.80,0.40,-0.30]])
col_sum=A.sum(axis=0)
row_mean=A.mean(axis=1)
max_value=A.max()
print("col_sum=",col_sum,"shape=",col_sum.shape)
print("row_mean=",row_mean,"shape=",row_mean.shape)
print("max_value=",max_value)


print("\n===== NP-E7 一次广播处理所有行 =====")
X=np.array([[0.80,-0.60,0.70],
            [-0.40,0.50,0.20]])
scale=np.array([1.50,0.50,-1.00])
bias=np.array([-0.10,0.20,0.30])
Y=X*scale+bias
print("Y=",Y,"shape=",Y.shape)


print("\n===== NP-E8 把两张图像组成batch =====")
a=np.array([[[0.20,-0.40],[0.70,0.10]],
            [[0.30,0.80],[-0.20,0.50]],
            [[-0.10,0.60],[0.40,-0.30]]])
b=a+0.10
batch=np.stack([a,b],axis=0)
blue_channel=batch[:,2,:,:]
batch_bhwc=batch.transpose(0,2,3,1)
print("batch.shape=",batch.shape)
print("blue_channel=",blue_channel,"shape=",blue_channel.shape)
print("batch_bhwc.shape=",batch_bhwc.shape)


print("\n===== NP-M1 批量计算夹角 =====")
def angles_to(query,X):
    query=np.asarray(query,dtype=float)
    X=np.asarray(X,dtype=float)
    if query.ndim!=1 or X.ndim!=2 or X.shape[1]!=query.shape[0]:
        raise ValueError("query与X的shape不符合要求")
    query_length=np.linalg.norm(query)
    if query_length==0:
        raise ValueError("query不能为零向量")
    row_lengths=np.linalg.norm(X,axis=1)
    angles=np.full(X.shape[0],np.nan,dtype=float)
    valid=row_lengths!=0
    cosine=(X[valid]@query)/(row_lengths[valid]*query_length)
    cosine=np.clip(cosine,-1.0,1.0)
    angles[valid]=np.degrees(np.arccos(cosine))
    return angles

query=np.array([0.60,-0.80,0.00])
X=np.array([[0.30,-0.40,0.00],
            [-0.60,0.80,0.00],
            [0.80,0.60,0.00],
            [0.00,0.00,0.00]])
result=angles_to(query,X)
print("夹角为：",result,"shape=",result.shape)


print("\n===== NP-M2 余弦相似度Top-k检索 =====")
def top_k_cosine(query,gallery,k):
    query=np.asarray(query,dtype=float)
    gallery=np.asarray(gallery,dtype=float)
    if query.ndim!=1 or gallery.ndim!=2:
        raise ValueError("query必须为一维，gallery必须为二维")
    if gallery.shape[1]!=query.shape[0]:
        raise ValueError("gallery的每一行必须与query维度相同")
    query_length=np.linalg.norm(query)
    if query_length==0:
        raise ValueError("query不能为零向量")
    if not isinstance(k,(int,np.integer)) or k<=0:
        raise ValueError("k必须为正整数")
    gallery_length=np.linalg.norm(gallery,axis=1)
    valid=gallery_length!=0
    valid_indices=np.flatnonzero(valid)
    if k>len(valid_indices):
        raise ValueError("k不能大于gallery中非零向量的数量")
    similarity=(gallery[valid]@query)/(gallery_length[valid]*query_length)
    similarity=np.clip(similarity,-1.0,1.0)
    order=np.argsort(-similarity,kind="stable")[:k]
    return valid_indices[order],similarity[order]

query=np.array([0.60,-0.80,0.00])
gallery=np.array([[0.60,-0.80,0.00],
                  [0.30,-0.40,0.00],
                  [0.80,0.60,0.00],
                  [-0.60,0.80,0.00],
                  [0.00,0.00,0.00]])
ids=np.array(["img_a","img_b","img_c","img_d","img_zero"])
indices,similarity=top_k_cosine(query,gallery,3)
print("选中的行索引=",indices)
print("选中的相似度=",similarity)
print("选中的ID=",ids[indices])


print("\n===== NP-M3 所有点对之间的距离矩阵 =====")
A=np.array([[0.20,-0.40],
            [0.80,0.30],
            [-0.50,0.70]])
B=np.array([[0.10,0.20],
            [-0.60,-0.20]])
dif=A[:,None,:]-B[None,:,:]
D=np.sqrt(np.sum(dif**2,axis=2))
nearest_index=np.argmin(D,axis=1)
print("dif.shape=",dif.shape)
print("D=",D,"shape=",D.shape)
print("nearest_index=",nearest_index)


print("\n===== NP-M4 小型神经网络前向计算 =====")
X=np.array([[0.80,-0.60,0.70],
            [-0.40,-0.50,0.90]])
W1=np.array([[-0.10,0.20,-0.80],
             [0.30,-0.50,0.10]])
b1=np.array([-0.15,0.05])
W2=np.array([[0.20,-0.60],
             [-0.80,0.52]])
b2=np.array([0.10,-0.20])
Z=X@W1.T+b1
H=np.maximum(Z,0)
Y=H@W2.T+b2
print("Z=",Z,"shape=",Z.shape)
print("H=",H,"shape=",H.shape)
print("Y=",Y,"shape=",Y.shape)
