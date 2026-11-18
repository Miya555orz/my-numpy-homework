"""Student implementation of the final project's image encoder.

Complete TODO-01 through TODO-06 using NumPy. Model parameters are supplied.
The public classes retain the final project's interface. helpers.py is provided.
"""
import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector


class MultiHeadAttention:
    def __init__(self, weights, prefix, heads):
        self.qkv_weight = weights[prefix + '.in_proj_weight']  # [3D,D]
        self.qkv_bias = weights[prefix + '.in_proj_bias']      # [3D]
        self.out_weight = weights[prefix + '.out_proj.weight']  # [D,D]
        self.out_bias = weights[prefix + '.out_proj.bias']
        self.heads = heads
        self.width = self.out_bias.shape[0]
        self.head_dim = self.width // heads

    def split_heads(self, x):
        """TODO-02. Input [B,N,D], return [B,H,N,D/H].

        In this image model D=768, H=12 and D/H=64. Use reshape/transpose.
        Research the head organization independently. Preserve token order.
        """
        #先把最后一维拆成heads个头，再把head轴放到token轴前面
        batch,tokens,_=x.shape
        x=x.reshape(batch,tokens,self.heads,self.head_dim)
        return x.transpose(0,2,1,3)

    def __call__(self, x, causal=False):
        # Given: stored linear weights use [out,in]. Parameters stay fixed.
        qkv = x @ self.qkv_weight.T + self.qkv_bias
        q = self.split_heads(qkv[:, :, :self.width])
        k = self.split_heads(qkv[:, :, self.width:2*self.width])
        v = self.split_heads(qkv[:, :, 2*self.width:])

        #交换k最后两个轴，使每个query都可以和所有key做点积
        k_transpose=k.transpose(0,1,3,2)
        scores=(q@k_transpose)/np.sqrt(np.float32(self.head_dim))

        # Given for later text-encoder compatibility. Images use causal=False.
        if causal:
            scores = causal_mask(scores)
        coefficients = attention_weights(scores)  # supplied numerical helper

        #用注意力系数混合value，再把12个head合并回768维
        context=coefficients@v
        batch,_,tokens,_=context.shape
        merged=context.transpose(0,2,1,3).reshape(batch,tokens,self.width)

        #权重按照[out,in]保存，所以矩阵乘法时使用.T转置
        return merged@self.out_weight.T+self.out_bias


class TransformerBlock:
    def __init__(self, weights, prefix, heads):
        self.norm1 = LayerNorm(weights[prefix+'.ln_1.weight'], weights[prefix+'.ln_1.bias'])
        self.norm2 = LayerNorm(weights[prefix+'.ln_2.weight'], weights[prefix+'.ln_2.bias'])
        self.attention = MultiHeadAttention(weights, prefix+'.attn', heads)
        self.fc1_weight = weights[prefix+'.mlp.c_fc.weight']  # [4D,D]
        self.fc1_bias = weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight = weights[prefix+'.mlp.c_proj.weight']  # [D,4D]
        self.fc2_bias = weights[prefix+'.mlp.c_proj.bias']
        self.gelu = GELU()

    def __call__(self, x, causal=False):
        # Given: normalize each token and calculate its attention update.
        update = self.attention(self.norm1(x), causal=causal)

        #第一次残差连接，保留原来的token信息
        residual=x+update

        # Given: the second Norm and the exact checkpoint activation helper.
        u = self.norm2(residual)

        #先升维到3072，经过GELU后再投影回768维
        hidden=self.gelu(u@self.fc1_weight.T+self.fc1_bias)
        mlp_update=hidden@self.fc2_weight.T+self.fc2_bias

        #第二次残差连接
        return residual+mlp_update


class ImageEncoder:
    def __init__(self, weights):
        self.patch_weight = weights['visual.conv1.weight'].reshape(768, 3072)
        self.cls = weights['visual.class_embedding'].reshape(1, 1, 768)
        self.position = weights['visual.positional_embedding']  # [50,768]
        self.pre_norm = LayerNorm(weights['visual.ln_pre.weight'], weights['visual.ln_pre.bias'])
        self.blocks = [TransformerBlock(weights, 'visual.transformer.resblocks.' + str(i), 12)
                       for i in range(12)]
        self.post_norm = LayerNorm(weights['visual.ln_post.weight'], weights['visual.ln_post.bias'])
        self.projection = weights['visual.proj']  # [768,512], already [in,out]

    def patchify(self, images):
        """TODO-01. [B,3,224,224] -> [B,49,3072].

        Visit patches left to right, then top to bottom. Within a patch, flatten
        C,H,W in that order. Each 32x32 RGB patch contributes 3*32*32 values.
        Use reshape/transpose. Do not resize the already-prepared input.
        """
        batch,channels,height,width=images.shape
        patch_size=32

        #先拆出7x7的patch网格和每个patch内部的32x32像素
        patches=images.reshape(batch,channels,height//patch_size,patch_size,
                               width//patch_size,patch_size)

        #调整为[B,网格行,网格列,C,patch高,patch宽]
        patches=patches.transpose(0,2,4,1,3,5)

        #把7x7网格变成49行，每个patch按C,H,W展开成3072个数
        return patches.reshape(batch,49,channels*patch_size*patch_size)

    def embed(self, images):
        patches = self.patchify(images)  # [B,49,3072]

        #patch_weight按照[out,in]保存，因此使用.T转置
        patch_tokens=patches@self.patch_weight.T

        # Given: one supplied CLS row per image.
        cls = np.tile(self.cls, (images.shape[0], 1, 1))

        #np.concatenate沿token轴把CLS放到49个patch前面
        tokens=np.concatenate([cls,patch_tokens],axis=1)

        #position的shape为[50,768]，会广播到batch中的每张图
        return tokens+self.position

    def encode_tokens(self, images):
        """Provided assembly of token input and the 12 blocks; [B,50,768]."""
        x = self.pre_norm(self.embed(images))
        for block in self.blocks:
            x = block(x)
        return x

    def __call__(self, images):
        """TODO-06. Return one normalized output token [B,512] per image."""
        tokens = self.encode_tokens(images)

        #第0行是CLS token，代表整张图像
        cls_token=tokens[:,0,:]
        cls_token=self.post_norm(cls_token)

        #visual.proj已经按照[in,out]保存，不需要转置
        output=cls_token@self.projection

        return unit_vector(output)
