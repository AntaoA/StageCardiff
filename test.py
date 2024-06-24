import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
import random
import networkx as nx
import pickle
import os



    
def get_device():
    return torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in sentence.split()]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence in batch:
           tokenized.append( tokenize(sentence, start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        
        x_token = self.batch_tokenize(x, start_token, end_token)
        x_emb = self.embedding(x_token)
        pos = self.position_encoder().to(get_device())
        x_out = self.dropout(x_emb + pos)
        return x_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out

  
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, self_attention_mask):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)
        return x
    
class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)
        return x

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):
        batch_size, sequence_length, d_model = x.size() # in practice, this is the same for both languages...so we can technically combine with normal attention
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask) # We don't need the mask for cross attention, removing in outer function!
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        out = self.linear_layer(values)
        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNormalization(parameters_shape=[d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):
        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        return y

class Decoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        return y


class Transformer(nn.Module):
    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                fr_vocab_size,
                english_to_index,
                francais_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        super().__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, english_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, francais_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, fr_vocab_size)
        self.device = torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False,
                dec_end_token=False):
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out
    
    
    
    

# Generated this by filtering Appendix code

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'

chemin = "grail-master/data/fb237_v4/"
#chemin = "FB15K-237/Data/"





if os.path.exists(chemin + 'index.pickle'):
    with open(chemin + 'index.pickle', 'rb') as f:
        index_to_rel, rel_to_index, rel_vocab_size = pickle.load(f)
else:
    relations = []

    relations.append(START_TOKEN)
    relations.append(END_TOKEN)
    relations.append(PAD_TOKEN)


    lines = []

    with open(chemin + 'list_rel.txt', 'r') as file:
        lines += file.readlines()    
    


    # Ajouter les relation et les relations inverses
    for line in lines:
        line = line.strip()
        relations.append(line)
        relations.append(line + '-1')

    
    rel_vocab_size = len(relations)

    index_to_rel = {k:v.strip() for k,v in enumerate(relations)}
    rel_to_index = {v.strip():k for k,v in enumerate(relations)}


    with open(chemin + 'index.pickle', 'wb') as f:
        pickle.dump((index_to_rel, rel_to_index, len(relations)), f)




if os.path.exists(chemin + 'graphe_train.pickle'):
    with open(chemin + 'graphe_train.pickle', 'rb') as f:
        G = pickle.load(f)
else:
    G = nx.MultiDiGraph()
    # Ajout des noeuds et des arêtes au graphe    
    with open(chemin + 'train.txt', 'r') as file:
        for line in file:
            n1, r, n2 = line.strip().split('\t')
            G.add_node(n1)
            G.add_node(n2)
            G.add_edge(n1, n2, relation = r)
    with open(chemin + 'graphe_train.pickle', 'wb') as f:
        pickle.dump(G, f)



def random_walk(start_node, end_node, relation, max_length, graph=G, alpha=1.0, rti=rel_to_index, S=START_TOKEN, P=PAD_TOKEN, E=END_TOKEN):
    path = [S]
    current_node = start_node
    trouve = False
    for i in range(max_length-2):
        neighbors = list(graph.successors(current_node))

        if not neighbors:
            break
        
        def comp(a):
            if a <= (max_length - 3 - i):
                return 1
            else:
                return np.inf
        
        # Calculer les probabilités pour choisir le prochain nœud
        distances = np.array([comp(nx.shortest_path_length(graph, node, end_node)) if nx.has_path(graph, node, end_node) else np.inf for node in neighbors])

        weights = np.exp(-alpha * distances)
        probabilities = weights / np.sum(weights)

        # Choisir le prochain noeud en fonction des probabilités calculées
        next_node = random.choices(neighbors, weights=probabilities)[0]

        # next_node = random.choices(neighbors)[0]        
        d = graph.get_edge_data(current_node, next_node)
        r = d[random.randint(0, (len(d)-1))]['relation']
        if i == 0:
            if r == relation:
                break
        if current_node == start_node and next_node == end_node and r == relation:
            break
        path.append(r)
        current_node = next_node
        if current_node == end_node:
            trouve = True
            path.append(E)
            path += [P] * (max_length - i - 3)
            break
    if not trouve:
        path = []
    return path

d_model = 512
batch_size = 10
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
len_vocab = len(rel_to_index)
max_sequence_length = 6
nb_paths = 3000
learning_rate = 1e-4

# création des données de train

if os.path.exists(chemin + 'list_path.pickle'):
    with open(chemin + 'list_path.pickle', 'rb') as f:
        rel_src, rel_tgt = pickle.load(f)
else:
    rel_src = []
    rel_tgt = []
    
    i = 0
    
    while i < nb_paths:
        edge = random.choice(list(G.edges(data=True)))
        node1, node2, r = edge[0], edge[1], edge[2]['relation']
        path = random_walk(node1, node2, r, 6, alpha=2)
        if not path == []:
            print(i)
            i = i+1
            rel_src.append(r)
            rel_tgt.append(' '.join(path))
        
    with open(chemin + 'list_path.pickle', 'wb') as f:
        pickle.dump((rel_src, rel_tgt), f)


    
NEG_INFTY = -1e9



transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          len_vocab,
                          rel_to_index,
                          rel_to_index,
                          START_TOKEN, 
                          END_TOKEN, 
                          PAD_TOKEN)    
    
    
from torch.utils.data import Dataset, DataLoader
class TextDataset(Dataset):
    
    def __init__(self, nodes_input, relation_sequences):
        self.nodes_input = nodes_input
        self.relation_sequences = relation_sequences

    def __len__(self):
        return len(self.nodes_input)

    def __getitem__(self, idx):
        return self.nodes_input[idx], self.relation_sequences[idx]
    
    def collate_fn(self, batch):
        nodes_input, relation_sequences = zip(*batch)

        # Remplir relation_sequences pour assurer une taille constante
        max_rel_seq_len = max(len(rel_seq) for rel_seq in relation_sequences)
        padded_relation_sequences = [rel_seq + ['<PAD>'] * (max_rel_seq_len - len(rel_seq)) for rel_seq in relation_sequences]

        return nodes_input, padded_relation_sequences
        
dataset = TextDataset(rel_src, rel_tgt)

train_loader = DataLoader(dataset, batch_size)
iterator = iter(train_loader)


from torch import nn

criterian = nn.CrossEntropyLoss(ignore_index=rel_to_index[PAD_TOKEN],
                                reduction='none')

# When computing the loss, we are ignoring cases when the label is the padding token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), learning_rate)
device = torch.device('cpu')

def create_masks(src_batch, tgt_batch):
    num_sentences = len(src_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      src_sentence_length, tgt_sentence_length = len(src_batch[idx]), len(tgt_batch[idx])
      src_chars_to_padding_mask = np.arange(src_sentence_length + 1, max_sequence_length)
      tgt_chars_to_padding_mask = np.arange(tgt_sentence_length + 1, max_sequence_length)
      encoder_padding_mask[idx, :, src_chars_to_padding_mask] = True
      encoder_padding_mask[idx, src_chars_to_padding_mask, :] = True
      decoder_padding_mask_self_attention[idx, :, tgt_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, tgt_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, src_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, tgt_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask

transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10


with open("sortie.txt", "w") as f:
    for epoch in range(num_epochs):
        f.write(f"\n\nEpoch {epoch}\n")
        iterator = iter(train_loader)
        for batch_num, batch in enumerate(iterator):
            transformer.train()
            src_batch, tgt_batch = batch
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(src_batch, tgt_batch)
            optim.zero_grad()
            tgt_predictions = transformer(src_batch,
                                            tgt_batch,
                                            encoder_self_attention_mask.to(device), 
                                            decoder_self_attention_mask.to(device), 
                                            decoder_cross_attention_mask.to(device),
                                            enc_start_token=False,
                                            enc_end_token=False,
                                            dec_start_token=False,
                                            dec_end_token=False)
            labels = transformer.decoder.sentence_embedding.batch_tokenize(tgt_batch, start_token=False, end_token=False)
            loss = criterian(
                tgt_predictions.view(-1, len_vocab).to(device),
                labels.view(-1).to(device)
            ).to(device)
            valid_indicies = torch.where(labels.view(-1) == rel_to_index[PAD_TOKEN], False, True)
            loss = loss.sum() / valid_indicies.sum()
            loss.backward()
            optim.step()
            #train_losses.append(loss.item())
            if batch_num % 100 == 0:
                f.write(f"Iteration {batch_num} : {loss.item()}\n\n")
                f.write(f"English: {src_batch[0]}\n")
                f.write(f"francais Translation: {tgt_batch[0]}\n\n")
                tgt_sentence_predicted = torch.argmax(tgt_predictions[0], axis=1)
                predicted_sentence = ""
                for idx in tgt_sentence_predicted:
                    predicted_sentence += " " + index_to_rel[idx.item()]
                    if idx == rel_to_index[END_TOKEN]:
                        break
                f.write(f"francais Prediction: {predicted_sentence}\n\n")


                transformer.eval()
                tgt_sentence = ("",)
                src_sentence = ("/award/hall_of_fame/inductees./award/hall_of_fame_induction/inductee",)
                for word_counter in range(max_sequence_length):
                    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(src_sentence, tgt_sentence)
                    predictions = transformer(src_sentence,
                                            tgt_sentence,
                                            encoder_self_attention_mask.to(device), 
                                            decoder_self_attention_mask.to(device),
                                            decoder_cross_attention_mask.to(device),
                                            enc_start_token=False,
                                            enc_end_token=False,
                                            dec_start_token=False,
                                            dec_end_token=False)
                    next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                    next_token_index = torch.argmax(next_token_prob_distribution).item()
                    next_token = index_to_rel[next_token_index]
                    tgt_sentence = (tgt_sentence[0] + " " + next_token, )
                    if next_token == END_TOKEN:
                        break
                
                f.write(f"Evaluation translation (/award/hall_of_fame/inductees./award/hall_of_fame_induction/inductee) : {tgt_sentence}\n")
                f.write("-------------------------------------------\n\n")

    transformer.eval()
    def translate(src_sentence):
        src_sentence = (src_sentence,)
        tgt_sentence = ("",)
        for word_counter in range(max_sequence_length):
            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(src_sentence, tgt_sentence)
            predictions = transformer(src_sentence,
                                    tgt_sentence,
                                    encoder_self_attention_mask.to(device), 
                                    decoder_self_attention_mask.to(device), 
                                    decoder_cross_attention_mask.to(device),
                                    enc_start_token=False,
                                    enc_end_token=False,
                                    dec_start_token=True,
                                    dec_end_token=False)
            next_token_prob_distribution = predictions[0][word_counter]
            next_token_index = torch.argmax(next_token_prob_distribution).item()
            next_token = rel_to_index[next_token_index]
            tgt_sentence = (tgt_sentence[0] + next_token, )
            if next_token == END_TOKEN:
                break
        return tgt_sentence[0]
            
