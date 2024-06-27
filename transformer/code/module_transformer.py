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


def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    return mask


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        """
        :param max_len: Input length sequence.
        :param d_model: Embedding dimension.
        :param dropout: Dropout value (default=0.1)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        """
        Inputs of forward function
        :param x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
    
class TextGen(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, sequence_length):
        super(TextGen, self).__init__()
        self.pos_encoder = PositionalEncoding(max_len=sequence_length, d_model=embed_dim)
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(0.2)
        
    # Positional encoding is required. Else the model does not learn.

    def forward(self, x):
        emb = self.emb(x)
        
        # Generate input sequence mask with shape (SEQUENCE_LENGTH, SEQUENCE_LENGTH)
        input_mask = generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        x = self.pos_encoder(emb)
        x = self.decoder(x, memory=x, tgt_mask=input_mask, memory_mask=input_mask)
        x = self.dropout(x)
        out = self.linear(x)
        return out
