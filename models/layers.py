import numpy as np
import torch
import torch.nn as nn

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1,is_post_ln=True):
        super(FFN, self).__init__()
        self.is_post_ln = is_post_ln
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x):
        if self.is_post_ln:
            # FFN先计算，再LayerNorm加残差
            residual = x
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.layernorm(x + residual)
        else:
            # 先LayerNorm，再FFN加残差
            residual = x
            x = self.layernorm(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = x + residual
        return x
        

class Multi_Head_Attention(nn.Module):
    def __init__(self,d_model,n_heads,dropout=0.1):
        super(Multi_Head_Attention,self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_model // n_heads
        self.q = nn.Linear(d_model,d_model)
        self.k = nn.Linear(d_model,d_model)
        self.v = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None):
        B, T_q, _ = q.shape
        T_k = k.shape[1]
        T_v = v.shape[1]

        Q = self.q(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k(k).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v(v).view(B, T_v, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_k ** 0.5)  # (B, n_heads, T_q, T_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        atten = self.softmax(scores)
        atten = self.dropout(atten)

        out = torch.matmul(atten, V)  # (B, n_heads, T_q, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.out_proj(out)
        return out


