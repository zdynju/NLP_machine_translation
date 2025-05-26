import torch
from torch import nn

from models.embedding.token_embedding import TokenEmbeddings
from models.embedding.positional_encoding import PositionalEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, max_len, device, drop_prob):
        super(TransformerEmbedding, self).__init__()
        self.d_model = d_model
        self.tok_emb = TokenEmbeddings(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        # x: [batch_size, seq_len]
        tok_emb = self.tok_emb(x) * (self.d_model ** 0.5)  # [batch_size, seq_len, d_model]
        pos_emb = self.pos_emb(x)  # [seq_len, d_model]
        return self.drop_out(tok_emb + pos_emb)  # [batch_size, seq_len, d_model]

class NN_Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward=2048, src_vocab_size=None, tgt_vocab_size=None, dropout=0.1, device='cuda'):
        super(NN_Transformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.device = device
        self.src_embedding = TransformerEmbedding(d_model, src_vocab_size, 256, device, dropout)
        self.tgt_embedding = TransformerEmbedding(d_model, tgt_vocab_size, 256, device, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model

    def generate_padding_mask(self, seq, pad_idx):
        return (seq == pad_idx).to(torch.float)

    def forward(self, src, tgt):
        # 保留原始 token id 以生成 mask
        src_input = src
        tgt_input = tgt

        src_key_padding_mask = self.generate_padding_mask(src_input, pad_idx=0)
        tgt_key_padding_mask = self.generate_padding_mask(tgt_input, pad_idx=0)
        memory_key_padding_mask = src_key_padding_mask

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)

        # 嵌入
        src = self.src_embedding(src_input)
        tgt = self.tgt_embedding(tgt_input)

        output = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_mask=None,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(output)

    def encode(self, src):
        src_input = src
        src_key_padding_mask = self.generate_padding_mask(src_input, pad_idx=0)
        src = self.src_embedding(src_input)
        memory = self.transformer.encoder(
            src=src,
            src_key_padding_mask=src_key_padding_mask
        )
        return memory, src_key_padding_mask

    def decode(self, tgt, memory, memory_key_padding_mask):
        tgt_input = tgt
        tgt_key_padding_mask = self.generate_padding_mask(tgt_input, pad_idx=0)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(self.device)
        tgt = self.tgt_embedding(tgt_input)
        output = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        return self.generator(output)
