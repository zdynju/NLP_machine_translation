import torch
import torch.nn as nn
from models.layers import FFN, Multi_Head_Attention
from models.embedding.transformer_embedding import TransformerEmbedding

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, dropout=0.1, is_post_ln=True):
        super(EncoderBlock, self).__init__()
        self.is_post_ln = is_post_ln

        self.attention = Multi_Head_Attention(d_model, n_heads, dropout)
        self.ffn = FFN(d_model, ffn_hidden, dropout)

        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        if self.is_post_ln:
            # Post-LN attention
            _x = x
            x = self.attention(q=x, k=x, v=x, mask=mask)
            x = self.dropout1(x)
            x = _x + x
            x = self.layernorm1(x)

            # Post-LN FFN
            _x = x
            x = self.ffn(x)
            x = self.dropout2(x)
            x = _x + x
            x = self.layernorm2(x)
        else:
            # Pre-LN attention
            _x = self.layernorm1(x)
            x = self.attention(q=_x, k=_x, v=_x, mask=mask)
            x = self.dropout1(x)
            x = _x + x  # 注意：此时 _x 已是 layernorm 之后的

            # Pre-LN FFN
            _x = self.layernorm2(x)
            x = self.ffn(_x)
            x = self.dropout2(x)
            x = _x + x

        return x


class Encoder(nn.Module):
    def __init__(self, n_blocks, d_model, n_heads, ffn_hidden, vocal_size, max_len,
                 dropout=0.1, device='cpu', is_post_ln=True):
        super(Encoder, self).__init__()
        self.device = device
        self.embedding = TransformerEmbedding(d_model, vocal_size, max_len, device=device, drop_prob=dropout)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, ffn_hidden, dropout, is_post_ln=is_post_ln) for _ in range(n_blocks)
        )

    def forward(self, x, mask=None):
        x = x.to(self.device)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x, mask)
        return x
