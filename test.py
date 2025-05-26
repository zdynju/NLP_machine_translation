import torch
from torch.utils.data import DataLoader
from dataset import TranslationDataset
import torch.optim as optim
import torch.nn as nn
from models.transformer import Transformer
from utils import get_bleu,idx_to_word
import time
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import json
from tqdm import tqdm
from inference import translation_batch

with open('data/cmn-eng-simple/word2int_en.json', 'r', encoding='utf-8') as f:
    src_vocab = json.load(f)
with open('data/cmn-eng-simple/word2int_cn.json', 'r', encoding='utf-8') as f:
    tgt_vocab = json.load(f)
with open('data/cmn-eng-simple/int2word_cn.json', 'r', encoding='utf-8') as f:
    r_tgt_vocab = json.load(f)
    
    
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    # pad_sequence 要求输入是 list of tensors，每个 tensor 是 [seq_len]
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<PAD>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<PAD>'])
    
    return src_batch, tgt_batch


def test(model,dataloader,device,max_len):
    model.eval()
    batch_bleu = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch[0].to(device)
            tgt = batch[1].to(device)
            output = translation_batch(src,model,max_len,src_vocab,tgt_vocab,device,is_beam_search=True)
            total_bleu = []                       # 输入数据设备
            for j,sentence in enumerate(output):
                trg_words = idx_to_word(batch[1][j][1:], r_tgt_vocab)
                output_words = sentence
                # print(trg_words,'KKKKKK',sentence)
                bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                total_bleu.append(bleu)
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return batch_bleu  



test_dataset = TranslationDataset('data/cmn-eng-simple/testing.txt', src_vocab, tgt_vocab)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,collate_fn=collate_fn)

model = Transformer(encoder_num=6,decoder_num=6,d_model=256,hidden=1024,n_head=8,drop_prob=0.2,max_len=256,enc_vocal_size=len(src_vocab), dec_vocal_size=len(tgt_vocab),device='cuda').to('cuda')
model.load_state_dict(torch.load('/root/machine_translation/saved/model-1.947160892188549.pth'))

bleu = test(model,test_loader,'cuda',125)
print(bleu)
