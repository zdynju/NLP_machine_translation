from models.layers import FFN, Multi_Head_Attention
import torch
import torch.nn as nn
from models.embedding.transformer_embedding import TransformerEmbedding

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_hidden, dropout=0.1, is_post_ln=True):
        super(DecoderBlock, self).__init__()
        self.is_post_ln = is_post_ln

        self.attention1 = Multi_Head_Attention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(d_model)

        self.attention2 = Multi_Head_Attention(d_model, n_heads, dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(d_model)

        self.ffn = FFN(d_model, ffn_hidden, dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.layernorm3 = nn.LayerNorm(d_model)

    def forward(self, enc, dec, src_mask, tar_mask):
        if self.is_post_ln:
            # Post-LN结构
            _x = dec
            x = self.attention1(q=dec, k=dec, v=dec, mask=tar_mask)
            x = self.dropout1(x)
            x = _x + x
            x = self.layernorm1(x)

            _x = x
            x = self.attention2(q=x, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = _x + x
            x = self.layernorm2(x)

            _x = x
            x = self.ffn(x)
            x = self.dropout3(x)
            x = _x + x
            x = self.layernorm3(x)

        else:
            # Pre-LN结构
            _x = dec
            dec_ln = self.layernorm1(dec)
            x = self.attention1(q=dec_ln, k=dec_ln, v=dec_ln, mask=tar_mask)
            x = self.dropout1(x)
            x = _x + x

            _x = x
            x_ln = self.layernorm2(x)
            x = self.attention2(q=x_ln, k=enc, v=enc, mask=src_mask)
            x = self.dropout2(x)
            x = _x + x

            _x = x
            x_ln = self.layernorm3(x)
            x = self.ffn(x_ln)
            x = self.dropout3(x)
            x = _x + x

        return x


class Decoder(nn.Module):
    def __init__(self, n_blocks, d_model, n_heads, ffn_hidden, vocal_size, max_len,
                 dropout=0.1, device='cpu', is_post_ln=True):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = TransformerEmbedding(d_model, vocal_size, max_len, device=device, drop_prob=dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(d_model, n_heads, ffn_hidden, dropout, is_post_ln=is_post_ln) for _ in range(n_blocks)
        )

    def forward(self, x, enc, src_mask=None, tar_mask=None):
        x = x.to(self.device)
        enc = enc.to(self.device)
        x = self.embedding(x)
        for block in self.blocks:
            x = block(enc, x, src_mask, tar_mask)
        return x
