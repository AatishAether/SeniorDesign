import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.onnx
import math

# Positional encoding is modeled after the functions of sin and cos of (k/(n^((2i)/d))). k represents the position of the word in the sentence, n is set to 10,000 (I don't know
# what n is), d is the dimensionality of the model (which the paper uses 512), and i represents an index which refers to the individual dimensions within the embedding.

# Positional encodings are generated in excess so that they can be used if needed only.
def PosEnc(l, dim):
    # Generate a tensor of zeros to hold the sin and cos values for embeddings.
    posEnc = torch.zeros(l, dim)
    # Calculate both the even and the odd encodings using sin for even and cos for odd.
    evens = torch.arange(0, dim, 2).float()
    evens_denom = torch.pow(10000, evens/dim)
    odds = torch.arange(1, dim, 2)
    odds_denom = torch.pow(10000, (odds - 1)/dim)

    position = torch.arange(l, dtype = torch.float).reshape(l, 1)
    evens_PEnc = torch.sin(position/evens_denom)
    print("Evens: ", evens_PEnc)
    odds_PEnc = torch.cos(position/odds_denom)

    # Fill tensor with values.
    posEnc = torch.stack([evens_PEnc, odds_PEnc], dim=2)
    print("PE: ", posEnc)
    # Once the matrix is populated, which I believe should take the form of a triangular matrix, it may be beneficial to do
    # a softmax. If so, the top triangle should be populated with -inf.

    # Flatten and return
    return torch.flatten(posEnc, start_dim = 1, end_dim = 2)

# The positional encoder is called here with a maxmimum sequence length of 200 tokens. This is also called with a model
# dimensionality of 512, which is what the size should be throughout much of the transformer.
print("Flattened PE: ", PosEnc(200, 20))

# Attention, single headed
def Att(q, k, v, mask=None):
    dim_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2))/math.sqrt(dim_k)
    if mask is not None:
        scaled += mask
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention 

# n_heads = 8
class multiHA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        # Take example of 512
        self.dim = dim
        # Take example of 8
        self.num_heads = num_heads
        # This would then be 64
        self.h_dim = dim // num_heads
        # This would then operate on a matrix of dimensions (512 x 1536)
        self.qkv_l = nn.Linear(dim, 3 * dim)
        # This would then operate on a matrix of size (512 x 512)
        self.lin_l = nn.Linear(dim, dim)

    def forward(self, x, mask):
        # In given example above this would be 30 x 200 x 512.
        bat_s, seq_l, dim = x.size()
        # Run x through qkv_l which will return dims of 30 x 200 x 1536.
        # Reshape vector into bat_s x seq_l x num_heads x 3(dim // num_heads).
        # Reshaping adds the number of heads (8) and then will divide 1536 / 8.
        qkv = self.qkv_l(x).reshape(bat_s, seq_l, self.num_heads, 3 * self.h_dim)
        # Prepare vector to be split into a query, key, and vector.
        # bat_s x num_heads x seq_l x 3(dim // num_heads)
        # 30 x 8 x 200 x 192
        qkv = qkv.permute(0, 2, 1, 3)
        # Split each vector into a query, key, and vector.
        # bat_s x num_heads x seq_l x 3(dim // num_heads)
        # 30 x 8 x 200 x 64
        q, k, v = qkv.chunk(3, dim = -1)
        # Perform attention on a head given q, k, and v.
        # Will return att: 30 x 8 x 200 x 200, vals: 30 x 8 x 200 x 64
        vals, att = Att(q, k, v, mask)
        # Restore vectors into proper order and reshape
        # bat_s x seq_l x num_heads * h_dim
        # 30 x 200 x 8(64)
        vals = vals.permute(0, 2, 1, 3).reshape(bat_s, seq_l, self.num_heads * self.h_dim)
        # Caca in, caca out
        out = self.lin_l(vals)
        return out


class LayerNorm(nn.Module):
    def __init__(self, param_shape, eps = 1e-5):
        super().__init__()
        # Param_shape may be 512
        self.param_shape = param_shape
        self.eps = eps
        # Will return a 512 dimension vector.
        self.gamma = nn.Parameter(torch.ones(param_shape))
        # Will return a 512 dimension vector.
        self.beta = nn.Parameter(torch.zeros(param_shape))

    def forward(self, ins):
        # ins: bat_s x seq_l x dim
        dims = [-(i + 1) for i in range(len(self.param_shape))]
        avg = ins.mean(dim=dims, keepdim=True)
        var = ((ins - avg) ** 2).mean(dim=dims, keepdim=True)
        std_v = (var + self.eps).sqrt()
        y = (ins - avg) / std_v
        out = self.gamma * y + self.beta
        return out

class PosFFN(nn.Module):
    def __init__(self, dim, hid, d_prob = 0.1):
        super(PosFFN, self).__init__()
        self.lin1 = nn.Linear(dim, hid)
        self.lin2 = nn.Linear(hid, dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=d_prob)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.lin2(x)
        return x

class EncLayer(nn.Module):
    def __init__(self, dim, ffn_hid, num_heads, d_prob):
        super(EncLayer, self).__init__()
        self.attention = multiHA(dim=dim, num_heads=num_heads)
        self.norm1 = LayerNorm(param_shape=[dim])
        self.drop1 = nn.Dropout(p=d_prob)
        self.ffn1 = PosFFN(dim=dim, hid=ffn_hid, d_prob=d_prob)
        self.norm2 = LayerNorm(param_shape=[dim])
        self.drop2 = nn.Dropout(p=d_prob)

    def forward(self, x, SA_mask):
        x2 = x.clone()
        x = self.attention(x, mask=SA_mask)
        x = self.drop1(x)
        x = self.norm1(x + x2)
        x2 = x.clone()
        x = self.ffn1(x)
        x = self.drop2(x)
        x = self.norm2(x + x2)
        return x

class Encoder(nn.Module):
    def __init__(self, dim, ffn_hid, num_heads, d_prob, num_layers):
        super().__init__()
        self.layers = nn.Sequential(*[EncLayer(dim, ffn_hid, num_heads, d_prob) for _ in range(num_layers)])

    def forward(self, x):
        x = self.layers(x)
        return x


# class multiHA(nn.Module):
#     def __init__(self, n_head, dim, dim_k, dim_v):
#         self.n_head = n_head
#         self.dim_k = dim_k
#         self.dim_v = dim_v
#         self.w_qs = nn.Linear(dim, n_head * dim_k, bias = False)
#         self.w_ks = nn.Linear(dim, n_head * dim_k, bias=False)
#         self.w_vs = nn.Linear(dim, n_head * dim_v, bias=False)
#         self.fc = nn.Linear(n_head * dim_v, dim, bias=False)
#
#         self.attention = ScaledDotProductAttention(temperature = dim_k ** .5)
#
#         self.dropout = nn.Dropout(.1)
#         self.layerNorm = nn.LayerNorm(dim, eps = 1e-6)
#
#     def forward(self, q, v, k, mask=None):
#         dim_k, dim_v, n_head = self.dim_k, self.dim_v, self.n_head
#         s_b, l_q, l_k, l_v = q.size(0), q.size(1), k.size(1), v.size(1)
#         residual = q
#
#         q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
#         k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
#         v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
#
#         # Transpose for attention dot product: b x n x lq x dv
#         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
#
#         if mask is not None:
#             mask = mask.unsqueeze(1)   # For head axis broadcasting.
#
#         q, attn = self.attention(q, k, v, mask=mask)
#
#         # Transpose to move the head dimension back: b x lq x n x dv
#         # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
#         q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
#         q = self.dropout(self.fc(q))
#         q += residual
#
#         q = self.layer_norm(q)
#
#         return q, attn
    
