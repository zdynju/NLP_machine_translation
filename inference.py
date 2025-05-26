import torch
from models.transformer import Transformer
import json
from dataset import TranslationDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import tqdm
with open('data/cmn-eng-simple/word2int_en.json', 'r', encoding='utf-8') as f:
    src_vocab = json.load(f)
with open('data/cmn-eng-simple/word2int_cn.json', 'r', encoding='utf-8') as f:
    tgt_vocab = json.load(f)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    # pad_sequence 要求输入是 list of tensors，每个 tensor 是 [seq_len]
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<PAD>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<PAD>'])
    
    return src_batch, tgt_batch

    
test_dataset = TranslationDataset('data/cmn-eng-simple/testing.txt', src_vocab, tgt_vocab)


dataloader = DataLoader(test_dataset,batch_size=32,shuffle=True,collate_fn=collate_fn)

def greedy_decode(model,src,max_len,bos_idx,eos_idx,device):
    batch_size = src.size(0)
    ys = torch.ones(batch_size, 1).fill_(bos_idx).long().to(device)

    for i in range(max_len):
        out = model(src,ys)
        next_word = out[:,-1,:].argmax(-1).unsqueeze(1)
        ys = torch.concat([ys,next_word],dim=1)
        
        if (next_word == eos_idx).all():
            break
    
    return ys


# def beam_search_decode(
#     model, src, bos_idx, eos_idx, device, 
#     beam_size=4, maxlen=100, alpha=0.7, pad_idx=0
# ):
#     model.eval()
#     with torch.no_grad():
#         # src: [1, src_len]
#         src = src.to(device)

#         beams = [([bos_idx], 0.0)]  # 初始 beam: ([bos], score)

#         for _ in range(maxlen):
#             new_beams = []
#             for seq, score in beams:
#                 if seq[-1] == eos_idx:
#                     new_beams.append((seq, score))
#                     continue

#                 tgt_input = torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)  # [1, len]
#                 out = model(
#                     src=src,
#                     tgt=tgt_input
#                 )  # [1, len, vocab_size]

#                 log_probs = F.log_softmax(out[:, -1, :], dim=-1)  # 取最后一位输出 [1, vocab_size]
#                 topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)  # [1, beam_size]

#                 for i in range(beam_size):
#                     token = topk_tokens[0, i].item()
#                     log_p = topk_log_probs[0, i].item()
#                     new_seq = seq + [token]
#                     new_score = score + log_p
#                     new_beams.append((new_seq, new_score))

#             def length_penalty(length, alpha=alpha):
#                 return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

#             beams = sorted(
#                 new_beams,
#                 key=lambda x: x[1] / length_penalty(len(x[0]), alpha),
#                 reverse=True
#             )[:beam_size]

#             if all(seq[-1] == eos_idx for seq, _ in beams):
#                 break

#         return beams[0][0]  # 返回最佳序列（token id list）
def beam_search_decode(model, src, bos_idx, eos_idx, device, beam_size=4, maxlen=100, alpha=0.7):
    model.eval()
    with torch.no_grad():
        memory, src_mask = model.encode(src)
        beams = [([bos_idx], 0.0)]  # (token list, log-prob score)

        for _ in range(maxlen):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == eos_idx:
                    new_beams.append((seq, score))
                    continue
                seq_tensor = torch.tensor(seq).unsqueeze(0).to(device)  # shape: [1, len]
                logits = model.decode_step(seq_tensor, memory, src_mask)  # shape: [1, len, vocab_size]
                log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # shape: [1, vocab_size]
                topk_log_probs, topk_tokens = torch.topk(log_probs, beam_size)
                topk_log_probs = topk_log_probs.squeeze(0)  # shape: [beam_size]
                topk_tokens = topk_tokens.squeeze(0)

                for i in range(beam_size):
                    token = topk_tokens[i].item()
                    log_p = topk_log_probs[i].item()
                    new_seq = seq + [token]
                    new_score = score + log_p
                    new_beams.append((new_seq, new_score))

            def length_penalty(length, alpha=alpha):
                return ((5 + length) ** alpha) / ((5 + 1) ** alpha)

            beams = sorted(
                new_beams,
                key=lambda x: x[1] / length_penalty(len(x[0]), alpha),
                reverse=True
            )[:beam_size]

            if all(seq[-1] == eos_idx for seq, _ in beams):
                break

        return beams[0][0]
    
def translation_batch(src,model,max_len,src_vocab,tgt_vocab,device,is_beam_search=False):
    model.eval()   
    with torch.no_grad():
        bos_idx = tgt_vocab['<BOS>']
        eos_idx = tgt_vocab['<EOS>']
        if is_beam_search:
            output = beam_search_decode(model,src,bos_idx,eos_idx,device)
            output = torch.tensor(output).unsqueeze(0)
        else: 
            output = greedy_decode(model, src, max_len, bos_idx, eos_idx,device)
        results = []
        for sent in output:
            tokens = [list(tgt_vocab.keys())[list(tgt_vocab.values()).index(idx.item())] for idx in sent]
            tokens = tokens[1:]  # remove <bos>
            if '<EOS>' in tokens:
                tokens = tokens[:tokens.index('<EOS>')]
            results.append(' '.join(tokens))
        return results
