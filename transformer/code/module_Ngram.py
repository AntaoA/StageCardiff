import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset
from torch.nn import functional as F

START_TOKEN = '<START>'
END_TOKEN = '<END>'
SEP_TOKEN = '<SEP>'
PAD_TOKEN = '<PAD>'


import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, data, rel_to_int, n):
        self.data = data  # Liste de séquences de données de la forme "r SEP START c END"
        self.n = n  # Taille des n-grammes à générer
        self.ngrams = self.build_ngrams(rel_to_int)  # Construire les n-grammes à partir des données
    def __len__(self):
        return len(self.ngrams)
    
    def __getitem__(self, idx):
        return self.ngrams[idx]
    
    
    def build_ngrams(self, rel_to_int):
        ngrams = []
        for sequence in self.data:
            # Générer tous les n-grammes possibles de taille n à partir de la séquence
            sequence = [PAD_TOKEN] * (self.n - 1) + sequence
            for i in range(len(sequence) - self.n + 1):
                ngram = tuple(sequence[i:i + self.n])
                ngram = [rel_to_int[rel] for rel in ngram]
                ngrams.append(ngram)
        return ngrams
    
    
    
class NGramTextGen(nn.Module):
    def __init__(self, vocab_size, context_size, embed_dim, hidden_dim):
        super(NGramTextGen, self).__init__()
        self.n_word = vocab_size
        self.emb = nn.Embedding(self.n_word, embed_dim)
        self.linear1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.n_word)

    def forward(self, x):
        emb = self.emb(x)
        emb = emb.view(-1, emb.size(1) * emb.size(2))
        out = self.linear1(emb)
        out = F.relu(out)
        out = self.linear2(out)
        return out