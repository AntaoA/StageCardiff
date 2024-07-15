import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset

START_TOKEN = '<START>'
END_TOKEN = '<END>'
SEP_TOKEN = '<SEP>'
PAD_TOKEN = '<PAD>'

class TextDataset(Dataset):
    def __init__(self, samples, word_to_int):
        self.samples = samples
        self.word_to_int = word_to_int
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.LongTensor([self.word_to_int[word] for word in sample[:-1]])
        target_seq = torch.LongTensor([self.word_to_int[word] for word in sample[1:]])
        padding_mask = (input_seq != self.word_to_int[PAD_TOKEN])
        return input_seq, target_seq, padding_mask


class LSTMTextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, sequence_length):
        super(LSTMTextGen, self).__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
    # Positional encoding is required. Else the model does not learn.

    def forward(self, x):
        emb = self.emb(x)
        
        
        h_0 = torch.zeros(self.num_layers, emb.size(0), self.hidden_dim).to(emb.device)    
        c_0 = torch.zeros(self.num_layers, emb.size(0), self.hidden_dim).to(emb.device)
        out, _ = self.lstm(emb, (h_0, c_0))
        out = self.dropout(out)
        out = self.fc(out)
        return out
