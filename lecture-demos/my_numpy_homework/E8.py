import numpy as np
#把两张图像组成batch
a = np.array([
    [[0.20, -0.40],
     [0.70, 0.10]],

    [[0.30, 0.80],
     [-0.20, 0.50]],

    [[-0.10, 0.60],
     [0.40, -0.30]]
])
b=a+0.10
#np.stack()函数可以把多个数组沿着指定的轴堆叠起来

batch=np.stack([a,b],axis=0)
print("batch=", batch)
print("batch.shape=", batch.shape)

blue_channel=batch[:,2,:,:]
print("blue_channel=", blue_channel)
print("blue_channel.shape=", blue_channel.shape)

#transpose()函数可以交换数组的轴
batch_bhwc=batch.transpose(0,2,3,1)
print("batch_bhwc=", batch_bhwc)
print("batch_bhwc.shape=", batch_bhwc.shape)