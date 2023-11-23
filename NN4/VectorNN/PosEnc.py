import torch
import torch.nn as nn
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
    odds_PEnc = torch.cos(position/odds_denom)

    # Fill tensor with values.
    posEnc = torch.stack([evens_PEnc, odds_PEnc], dim=2)
    # Once the matrix is populated, which I believe should take the form of a triangular matrix, it may be beneficial to do
    # a softmax. If so, the top triangle should be populated with -inf.

    # Flatten and return
    return torch.flatten(posEnc, start_dim = 1, end_dim = 2)

# The positional encoder is called here with a maxmimum sequence length of 200 tokens. This is also called with a model
# dimensionality of 512, which is what the size should be throughout much of the transformer.
PosEnc(200, 512)


# n_heads = 8

class multiHA(nn.Module):
    def __init__(self, n_head, dim, dim_k, dim_v):
        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.w_qs = nn.Linear(dim, n_head * dim_k, bias = False)
        self.w_ks = nn.Linear(dim, n_head * dim_k, bias=False)
        self.w_vs = nn.Linear(dim, n_head * dim_v, bias=False)
        self.fc = nn.Linear(n_head * dim_v, dim, bias=False)

        self.attention = ScaledDotProductAttention(temperature = dim_k ** .5)

        self.dropout = nn.Dropout(.1)
        self.layerNorm = nn.LayerNorm(dim, eps = 1e-6)

    def forward(self, q, v, k, mask=None):
        dim_k, dim_v, n_head = self.dim_k, self.dim_v, self.n_head
        s_b, l_q, l_k, l_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn
    
