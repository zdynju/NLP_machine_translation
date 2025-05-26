import  numpy as np
import torch
import torch.nn as nn
from models.encoder import Encoder
from models.decoder import Decoder


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)  

    
class Transformer(nn.Module):
    def __init__(self,encoder_num,decoder_num,d_model,n_head,drop_prob,hidden,max_len,enc_vocal_size,dec_vocal_size,device,is_post_ln=True):
        super(Transformer,self).__init__()
        self.encoder = Encoder(n_blocks=encoder_num,d_model=d_model,
                                n_heads=n_head,dropout=drop_prob,ffn_hidden=hidden,vocal_size=enc_vocal_size,max_len=max_len,device=device,is_post_ln=is_post_ln)
        self.decoder = Decoder(n_blocks=decoder_num,d_model=d_model,
                                n_heads=n_head,dropout=drop_prob,ffn_hidden=hidden,vocal_size=dec_vocal_size,max_len=max_len,device=device,is_post_ln=is_post_ln)
        self.generator = Generator(d_model,dec_vocal_size)
        
        self.device = device
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
                
    def encode(self,src):
        src_mask = self.get_src_mask(src)
        memory = self.encoder(src, src_mask)
        return memory, src_mask

    def decode_step(self,tar,memory,src_mask):
        tar_mask = self.get_tar_mask(tar)
        dec_output = self.decoder(tar,memory,src_mask,tar_mask)
        logits = self.generator(dec_output)
        return logits
        
    def forward(self,src,tar):
        
        src_mask = self.get_src_mask(src)
        tar_mask = self.get_tar_mask(tar)
        # print('src_mask:',src_mask)
        # print('tar_mask:',tar_mask)
        enc_output = self.encoder(src,src_mask)
        dec_output = self.decoder(tar,enc_output,src_mask,tar_mask)
        logits = self.generator(dec_output)
        return logits
    
    def get_src_mask(self,src):
        # src:[batch_size,length]
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(self.device) 
        return src_mask
        # src_mask:[batch_size,1,1,length]
    def get_tar_mask(self,tar):

        length = tar.size(1)
        tar_sub_mask = torch.tril(torch.ones((length,length))).type(torch.float)
        tar_sub_mask = tar_sub_mask.unsqueeze(0).expand(tar.size(0),-1,-1).unsqueeze(1).to(self.device)

        tar_pad_mask = (tar != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        tar_mask = tar_sub_mask & tar_pad_mask
     
        return tar_mask
