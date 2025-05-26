import torch
from torch.utils.data import DataLoader
from dataset import TranslationDataset
import torch.optim as optim
import torch.nn as nn
from utils import get_bleu, idx_to_word
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from inference import translation_batch
from transformers import get_cosine_schedule_with_warmup
from torch.nn.utils.rnn import pad_sequence
from models.nn_transformer import NN_Transformer
# 读取词表
with open('data/cmn-eng-simple/word2int_en.json', 'r', encoding='utf-8') as f:
    src_vocab = json.load(f)
with open('data/cmn-eng-simple/word2int_cn.json', 'r', encoding='utf-8') as f:
    tgt_vocab = json.load(f)
with open('data/cmn-eng-simple/int2word_cn.json', 'r', encoding='utf-8') as f:
    r_tgt_vocab = json.load(f)

def collate_fn(batch):
    """数据批处理，填充序列至最大长度"""
    src_batch, tgt_batch = zip(*batch)
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_vocab['<PAD>'])
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_vocab['<PAD>'])
    return src_batch, tgt_batch

# 数据集和数据加载器
train_dataset = TranslationDataset('data/cmn-eng-simple/training.txt', src_vocab, tgt_vocab)
validation_dataset = TranslationDataset('data/cmn-eng-simple/validation.txt', src_vocab, tgt_vocab)
test_dataset = TranslationDataset('data/cmn-eng-simple/testing.txt', src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化 Transformer 模型（使用 PyTorch 自带的 nn.Transformer）
model = NN_Transformer(
    d_model=256,
    nhead=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=1024,
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    dropout=0.1,
    device=device
).to(device)

# 优化器和学习率调度器
optimizer = optim.AdamW(
    model.parameters(),
    lr=5e-4,
    betas=(0.9, 0.98),
    eps=1e-9,
    weight_decay=0.01
)

total_steps = len(train_loader) * 60  # 训练总步数：epoch数*每epoch批次数
warmup_steps = int(total_steps * 0.1)  # 前10%作为warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 损失函数，带标签平滑
criterion = nn.CrossEntropyLoss(
    ignore_index=tgt_vocab['<PAD>'],
    label_smoothing=0.1
)

def train(model, dataloader, optimizer, criterion, device, scheduler=None,
          src_pad_idx=0, tgt_pad_idx=0):
    model.train()
    epoch_loss = 0
    print('开始训练 epoch...')

    for batch in tqdm(dataloader):
        src = batch[0].to(device)  # [B, src_len]
        tgt = batch[1].to(device)  # [B, tgt_len]

        tgt_input = tgt[:, :-1]  # decoder 输入
        tgt_output = tgt[:, 1:]  # 目标输出

        optimizer.zero_grad()

        output = model(
            src=src,
            tgt=tgt_input
        )  # [B, tgt_len-1, vocab_size]

        output = output.reshape(-1, output.size(-1))
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"训练完成，平均损失：{avg_loss:.4f}")
    return avg_loss

def evaluate(model, dataloader, criterion, device, src_pad_idx=0, tgt_pad_idx=0):
    model.eval()
    epoch_loss = 0
    print('开始验证...')

    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch[0].to(device)
            tgt = batch[1].to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            output = model(
                src=src,
                tgt=tgt_input
            )
            output = output.reshape(-1, output.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"验证完成，平均损失：{avg_loss:.4f}")
    return avg_loss

def test(model, dataloader, device, max_len):
    """用 BLEU 评估模型性能"""
    model.eval()
    batch_bleu = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            src = batch[0].to(device)
            tgt = batch[1].to(device)

            # 生成翻译结果，translation_batch 需返回 list[str]，每个字符串是句子
            output_sentences = translation_batch(src, model, max_len, src_vocab, tgt_vocab, device)

            total_bleu = []
            for j, generated_sentence in enumerate(output_sentences):
                reference_ids = tgt[j][1:]  # 去除起始符号
                reference_tokens = idx_to_word(reference_ids, r_tgt_vocab)

                hypothesis_tokens = generated_sentence.strip().split()

                bleu = get_bleu(hypotheses=hypothesis_tokens, reference=reference_tokens.split())
                total_bleu.append(bleu)

            avg_batch_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(avg_batch_bleu)

    avg_bleu = sum(batch_bleu) / len(batch_bleu)
    print(f"测试集 BLEU 分数: {avg_bleu:.4f}")
    return avg_bleu

def run(num_epochs, best_loss, device, scheduler):
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device, scheduler,
                           src_pad_idx=src_vocab['<PAD>'], tgt_pad_idx=tgt_vocab['<PAD>'])
        valid_loss = evaluate(model, validation_loader, criterion, device,
                              src_pad_idx=src_vocab['<PAD>'], tgt_pad_idx=tgt_vocab['<PAD>'])

        train_losses.append(train_loss)
        val_losses.append(valid_loss)

        # 仅在训练15个epoch后保存表现更优模型
        if valid_loss < best_loss and epoch >= 15:
            best_loss = valid_loss
            torch.save(model.state_dict(), f'saved/model-{valid_loss:.4f}.pth')
            print(f"模型已保存，当前最佳验证损失: {valid_loss:.4f}")

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

    test_bleu = test(model, test_loader, device, max_len=256)
    print(f"测试 BLEU 分数：{test_bleu:.4f}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="训练损失")
    plt.plot(range(1, num_epochs + 1), val_losses, label="验证损失")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练与验证损失曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.close()


if __name__ == '__main__':
    run(num_epochs=60, best_loss=torch.inf, device=device, scheduler=scheduler)
