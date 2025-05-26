from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, file, src_vocab, tgt_vocab):
        self.sentences_pair = self.load_sentences(file)
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def load_sentences(self, filename):
        sentences = []
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('\t')
                if len(line) == 2:
                    sentences.append((line[0], line[1]))
        return sentences

    def __len__(self):
        return len(self.sentences_pair)

    def __getitem__(self, idx):
        src_sentence, tgt_sentence = self.sentences_pair[idx]
        # 在源和目标句子中添加 BOS 和 EOS
        source_sentences_with_bos_eos = ['<BOS>'] + src_sentence.split() + ['<EOS>']
        target_sentences_with_bos_eos = ['<BOS>'] + tgt_sentence.split() + ['<EOS>']
        
        # 将源语言和目标语言的句子转化为数字索引
        src_indices = [self.src_vocab.get(word, self.src_vocab.get('<UNK>', 0)) for word in source_sentences_with_bos_eos]
        tgt_indices = [self.tgt_vocab.get(word, self.tgt_vocab.get('<UNK>', 0)) for word in target_sentences_with_bos_eos]
        # 将索引转换为张量并返回
        return torch.tensor(src_indices), torch.tensor(tgt_indices)
