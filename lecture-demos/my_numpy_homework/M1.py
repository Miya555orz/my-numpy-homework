import numpy as np
def angles_to(query, X):
    #计算 query 与 X 每一行的夹角，单位为度
    #np.array 默认会复制输入数据，而 np.asarray 在输入已是 ndarray 且 dtype 不变时不会复制，而是返回原数组的引用。
    query=np.asarray(query,dtype=float)
    X=np.asarray(X,dtype=float)
    #向量长度(每一行的欧几里得距离)
    row_lengths=np.linalg.norm(X,axis=1)
    #全为nan的结果数组
    angles=np.full(X.shape[0], np.nan,dtype=float)
    #找出X中的非零向量
    valid=row_lengths!=0
    #X[valid] @ query 会一次计算每个有效行与 query 的点积
    #X[valid] 会只保留不为0的行
    cosine=(X[valid]@query)/(row_lengths[valid]*np.linalg.norm(query))
    #防止浮点数误差
    #>最大值返回最大值，最小值也是
    cosine=np.clip(cosine, -1.0, 1.0)
    #反余弦得到弧度再转角度
    angles[valid]=np.degrees(np.arccos(cosine))
    return angles


query = np.array([0.60, -0.80, 0.00])
X = np.array([
    [0.30, -0.40, 0.00],
    [-0.60, 0.80, 0.00],
    [0.80, 0.60, 0.00],
    [0.00, 0.00, 0.00]
])
result=angles_to(query, X)

print("夹角为：", result)
print("夹角的形状为：", result.shape)