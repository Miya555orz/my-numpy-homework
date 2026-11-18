import numpy as np
#返回与query最相似的k个向量的索引和相似度
def top_k_cosine(query,gallery,k):
    query=np.asarray(query,dtype=float)
    gallery=np.asarray(gallery,dtype=float)

    #检查query是否为1维，gallery是否为2维
    if query.ndim!=1:
        raise ValueError("query必须为一维向量")
    if gallery.ndim!=2:
        raise ValueError("gallery必须为二维向量")

    #gallery的每一行必须与query维度相同
    if gallery.shape[1]!=query.shape[0]:
        raise ValueError("gallery的每一行必须与query维度相同")

    query_length=np.linalg.norm(query)

    if query_length==0:
        raise ValueError("query不能为零向量")

    if not isinstance(k,(int,np.integer)) or k<=0:
        raise ValueError("k必须为正整数")
    #计算gallery中每一行的长度
    gallery_length=np.linalg.norm(gallery,axis=1)

    valid=gallery_length!=0

    valid_indices=np.flatnonzero(valid)

    if k>len(valid_indices):
        raise ValueError("k不能大于gallery中非零向量的数量")
    similarity=(gallery[valid]@query)/(gallery_length[valid]*query_length)

    similarity=np.clip(similarity,-1.0,1.0)

    order=np.argsort(-similarity,kind="stable")

    selected_order=order[:k]
    selected_indices=valid_indices[selected_order]
    selected_similarity=similarity[selected_order]

    return selected_indices,selected_similarity

query=np.array([0.60,-0.80,0.00])

gallery = np.array([
    [0.60, -0.80, 0.00],
    [0.30, -0.40, 0.00],
    [0.80, 0.60, 0.00],
    [-0.60, 0.80, 0.00],
    [0.00, 0.00, 0.00]
])

ids = np.array([
    "img_a",
    "img_b",
    "img_c",
    "img_d",
    "img_zero"
])

indices,similarity=top_k_cosine(query,gallery,3)

print("选中的行索引=", indices)
print("选中的相似度=", similarity)
print("选中的ID=", ids[indices])