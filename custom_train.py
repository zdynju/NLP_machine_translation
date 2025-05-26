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
from transformers import get_cosine_schedule_with_warmup



with open('data/cmn-eng-simple/word2int_en.json', 'r', encoding='utf-8') as f:
    src_vocab = json.load(f)
with open('data/cmn-eng-simple/word2int_cn.json', 'r', encoding='utf-8') as f:
    tgt_vocab = json.load(f)
with open('data/cmn-eng-simple/int2word_cn.json', 'r', encoding='utf-8') as f:
    r_tgt_vocab = json.load(f)
    
# print(r_tgt_vocab)
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<PAD>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<PAD>'])
    
    return src_batch, tgt_batch

train_dataset = TranslationDataset('data/cmn-eng-simple/training.txt', src_vocab, tgt_vocab)
test_dataset = TranslationDataset('data/cmn-eng-simple/testing.txt', src_vocab, tgt_vocab)
validation_dataset = TranslationDataset('data/cmn-eng-simple/validation.txt',src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True,collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,collate_fn=collate_fn)
validation_loader =DataLoader(validation_dataset, batch_size=32, shuffle=False,collate_fn=collate_fn)

model = Transformer(encoder_num=6,
                    decoder_num=6,
                    d_model=256,
                    hidden=1024,
                    n_head=8,
                    drop_prob=0.1,
                    max_len=64,
                    enc_vocal_size=len(src_vocab), 
                    dec_vocal_size=len(tgt_vocab),
                    device='cuda',
                    is_post_ln=False).to('cuda')


optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.98),  # 使用推荐的beta值
    eps=1e-9,           # 使用更小的epsilon
    weight_decay=0.01   # 增大权重衰减
)

criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_vocab['<PAD>'],
    label_smoothing=0.1  # 添加标签平滑
)
total_steps = len(train_loader) * 60
warmup_steps = int(total_steps * 0.1)  # 前10%步数进行warm-up

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def train(model,dataloader,optimizer,criterion,device,scheduler):
    model.train()
    epoch_loss = 0
    # batch_bleu = []
    print('train_epoch')
    # count = 0
    for batch in tqdm(dataloader):
        # count+=1
        src = batch[0].to(device)
        tar = batch[1].to(device)
        optimizer.zero_grad()
        dec_input = tar[:,:-1]
        enc_input = src
        output = model(enc_input,dec_input)
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        tar = tar[:, 1:].contiguous().view(-1)
     

        loss = criterion(output_reshape,tar)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model,dataloader,criterion,device):
    model.eval()
    epoch_loss = 0
    # batch_bleu = []
    print('evaluate_epoch')
    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch[0].to(device)
            tar = batch[1].to(device)
            dec_input = tar[:,:-1]
            enc_input = src
            output = model(enc_input,dec_input)
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            tar = tar[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, tar)
            
            epoch_loss += loss.item()

    return epoch_loss / len(dataloader)  


def test(model,dataloader,device,max_len):
    model.eval()
    batch_bleu = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch[0].to(device)
            tgt = batch[1].to(device)
            output = translation_batch(src,model,max_len,src_vocab,tgt_vocab,device)
            total_bleu = []
            for j,sentence in enumerate(output):
                trg_words = idx_to_word(batch[1][j][1:], r_tgt_vocab)
                output_words = sentence
                print('1',trg_words)
                print('2',output_words)
                bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                total_bleu.append(bleu)
            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)
    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return batch_bleu  


def run(epoch,best_loss,device,scheduler):
    train_losses, val_losses = [], []
    for step in range(epoch):
        train_loss = train(model, train_loader, optimizer, criterion,device,scheduler )
        valid_loss = evaluate(model, validation_loader, criterion,device)

        
        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        if valid_loss < best_loss and step > 15:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'saved/model-{0}.pth'.format(valid_loss))
            
        print(f"Epoch {step+1}/{epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}" ) #, BLEU: {bleu:.4f}")

    test_bleu = test(model,test_loader,'cuda',256)
    print(test_bleu)
    
    plt.figure(figsize=(10, 5))
    
    # 绘制训练损失和验证损失曲线
    plt.plot(range(1, epoch+1), train_losses, label="Train Loss")
    plt.plot(range(1, epoch+1), val_losses, label="Validation Loss")
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 显示图像
    plt.savefig("loss_curve.png") 



if __name__ == '__main__':
    run(epoch=60, best_loss=torch.inf,device='cuda',scheduler=scheduler)    
