
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
import random
import networkx as nx

from torch.utils.data import Dataset, DataLoader

    
def get_device():
    return torch.device('cuda')

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
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [self.language_to_index[token] for token in sentence]
            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PAD_TOKEN])
            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        tokenized = torch.stack(tokenized)
        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence
        
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        return x


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
                 PAD_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
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
        assert d_model % num_heads == 0, "d_model must be divisible bu num_heads"
        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y):
        batch_size, input_length, d_model = x.size() # in practice, this is the same for both languages...so we can technically combine with normal attention
        _, output_length, _ = y.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, input_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, output_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        values, attention = scaled_dot_product(q, k, v, None)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, output_length, d_model)
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
        y = self.encoder_decoder_attention(x, y)
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
                 PAD_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
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
                rel_vocab_size,
                nodes_to_index,
                rel_to_index,
                START_TOKEN, 
                END_TOKEN, 
                PAD_TOKEN
                ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, 2, nodes_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, rel_to_index, START_TOKEN, END_TOKEN, PAD_TOKEN)
        self.linear = nn.Linear(d_model, rel_vocab_size)
        self.device = torch.device('cuda')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, # We should make t his true
                dec_end_token=False): # x, y are batch of sentences
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)
        return out
    
    
# Generated this by filtering Appendix code

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'

with open('Data/list_noeuds.txt', 'r') as file:
    nodes = file.readlines()

with open('Data/list_rel.txt', 'r') as file:
    relations = file.readlines()
    
    
# On ajoute les tokens. Pas besoin de PAD dans pour les nodes car ils ne sont utile qu'en entrée et il y en a tout le temps 2, donc tout le temps la même taille.

nodes.append(START_TOKEN)
nodes.append(END_TOKEN)

relations.append(START_TOKEN)
relations.append(END_TOKEN)
relations.append(PAD_TOKEN)


index_to_rel = {k:v.strip() for k,v in enumerate(relations)}
rel_to_index = {v.strip():k for k,v in enumerate(relations)}

index_to_nodes = {k:v.strip() for k,v in enumerate(nodes)}
nodes_to_index = {v.strip():k for k,v in enumerate(nodes)}

with open('Data/train.txt', 'r') as file:
    list_triplets = file.readlines()

# création du graphe
G = nx.MultiDiGraph()
with open('Data/train.txt', 'r') as file:
    for line in file:
        n1, r, n2 = line.strip().split('\t')
        G.add_node(n1)
        G.add_node(n2)
        G.add_edge(n1, n2, relation = r)
        
# cette marche aléatoire est vraiment nulle

def random_walk(graph, start_node, end_node, max_length):
    path = [start_node]
    current_node = start_node
    
    for _ in range(max_length):
        neighbors = list(graph.successors(current_node))
        if not neighbors:
            break
        next_node = random.choice(neighbors)
        path.append(next_node)
        current_node = next_node
        if current_node == end_node:
            break
    
    return path


# déjà un peu mieux. Je met un poids sur chaque voisin, la longueur du plus petit chemin jusqu'à la fin
# pondéré en maxsoft * 2 car desfois il y a vraiment beaucoup de voisins

def biased_random_walk(graph, start_node, end_node, max_length, alpha=2.0):
    path = [start_node]
    current_node = start_node
    trouve = False
    
    for _ in range(max_length):
        neighbors = list(graph.successors(current_node))
        
        if not neighbors:
            break
        # Calculer les probabilités pour choisir le prochain nœud
        distances = np.array([nx.shortest_path_length(graph, node, end_node) if nx.has_path(graph, node, end_node) else np.inf for node in neighbors])
        
        weights = np.exp(-alpha * distances)
        probabilities = weights / np.sum(weights)

        # Choisir le prochain nœud en fonction des probabilités calculées
        next_node = random.choices(neighbors, weights=probabilities)[0]
        
        path.append(next_node)
        current_node = next_node
        
        if current_node == end_node:
            trouve = True
            break
    
    print(trouve)
    return path

paths = []
for i in range(3):
    print(i)
    match = False
    while not match:
        node1 = random.choice(list(G.nodes))
        node2 = random.choice(list(G.nodes))
        if nx.has_path(G, node1, node2):
            match = True
    paths.append(biased_random_walk(G, node1, node2, 100, 2))
    
    

# Extraire les séquences de relations entre deux nœuds

nodes_input = []
relation_sequences = []
for path in paths:
    path_rel = []
    
    # on rajoute le noeud de départ et d'arrivé dans nodes_input
    path_node = []
    path_node.append(path[0])
    path_node.append(path[-1])
    nodes_input.append(path_node)
    
    for i in range(1, len(path)-1):
        # Obtenir la relation entre les nœuds
        edge_data = G.get_edge_data(path[i-1], path[i])
        if edge_data:
            rel = [data['relation'] for key, data in edge_data.items()]
            path_rel.append(random.choice(rel)) #gérer le cas où il y a plusieurs relations entre deux mêmes noeuds
    relation_sequences.append(path_rel)

    
POURCENTILE = 97
print( f"{POURCENTILE}e pourcentile - longueur des chemins : {np.percentile([len(x) for x in relation_sequences], POURCENTILE)}" )

NEG_INFTY = -1e9


d_model = 512
batch_size = 10
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
max_sequence_length = 100
rel_vocab_size = len(relations)

transformer = Transformer(d_model, 
                          ffn_hidden,
                          num_heads, 
                          drop_prob, 
                          num_layers, 
                          max_sequence_length,
                          rel_vocab_size,
                          nodes_to_index,
                          rel_to_index,
                          START_TOKEN, 
                          END_TOKEN,
                          PAD_TOKEN)




class TextDataset(Dataset):
    
    # sert à stocker les données d'entrainement
    # __len__ renvoie le nombre de données
    # __getitem(i) retourne le i-ième couple ([n1, n2], chemin)
    
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
    
dataset = TextDataset(nodes_input, relation_sequences)

train_loader = DataLoader(dataset, batch_size, collate_fn = dataset.collate_fn)
iterator = iter(train_loader)



criterian = nn.CrossEntropyLoss(ignore_index=rel_to_index[PAD_TOKEN],
                                reduction='none')


# When computing the loss, we are ignoring cases when the label is the PAD token
for params in transformer.parameters():
    if params.dim() > 1:
        nn.init.xavier_uniform_(params)

optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = torch.device('cuda')


def create_masks(nodes_batch, rel_batch):
    num_nodes = len(nodes_batch)
    num_rel = len(rel_batch)
    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    encoder_PAD_mask = torch.full([num_nodes, 2, 2] , False)
    decoder_PAD_mask_self_attention = torch.full([num_rel, max_sequence_length, max_sequence_length] , False)
    decoder_PAD_mask_cross_attention = torch.full([num_rel, max_sequence_length, max_sequence_length] , False)

    for idx in range(min(num_nodes, num_rel)):
      node_sentence_length, rel_sentence_length = len(nodes_batch[idx]), len(rel_batch[idx])
      node_chars_to_PAD_mask = np.arange(node_sentence_length + 1, 2)
      rel_chars_to_PAD_mask = np.arange(rel_sentence_length + 1, max_sequence_length)
      encoder_PAD_mask[idx, :, node_chars_to_PAD_mask] = True
      encoder_PAD_mask[idx, node_chars_to_PAD_mask, :] = True
      decoder_PAD_mask_self_attention[idx, :, rel_chars_to_PAD_mask] = True
      decoder_PAD_mask_self_attention[idx, rel_chars_to_PAD_mask, :] = True
      decoder_PAD_mask_cross_attention[idx, :, node_chars_to_PAD_mask] = True
      decoder_PAD_mask_cross_attention[idx, rel_chars_to_PAD_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_PAD_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_PAD_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_PAD_mask_cross_attention, NEG_INFTY, 0)
    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10


for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        nodes_batch, rel_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(nodes_batch, rel_batch)
        optim.zero_grad()
        rel_predictions = transformer(nodes_batch,
                                         rel_batch,
                                         encoder_self_attention_mask.to(device), 
                                         decoder_self_attention_mask.to(device), 
                                         decoder_cross_attention_mask.to(device),
                                         enc_start_token=False,
                                         enc_end_token=False,
                                         dec_start_token=True,
                                         dec_end_token=True)
        labels = transformer.decoder.sentence_embedding.batch_tokenize(rel_batch, start_token=False, end_token=True)
        loss = criterian(
            rel_predictions.view(-1, rel_vocab_size).to(device),
            labels.view(-1).to(device)
        ).to(device)
        valid_indicies = torch.where(labels.view(-1) == rel_to_index[PAD_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"Nodes: {nodes_batch[0]}")
            print(f"Path: {rel_batch[0]}")
            rel_sentence_predicted = torch.argmax(rel_predictions[0], axis=1)
            predicted_sentence = ""
            for idx in rel_sentence_predicted:
              if idx == rel_to_index[END_TOKEN]:
                break
              predicted_sentence += index_to_rel[idx.item()]
            print(f"Path: {predicted_sentence}")
            print("-------------------------------------------")


# nodes : ['n1', 'n2']
transformer.eval()
def predict(nodes):
  relation_sequence = ([],)
  for i in range(max_sequence_length):
    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= create_masks(nodes, relation_sequence)
    predictions = transformer(nodes,
                              relation_sequence,
                              encoder_self_attention_mask.to(device), 
                              decoder_self_attention_mask.to(device), 
                              decoder_cross_attention_mask.to(device),
                              enc_start_token=False,
                              enc_end_token=False,
                              dec_start_token=True,
                              dec_end_token=False)
    next_token_prob_distribution = predictions[0][i]
    next_token_index = torch.argmax(next_token_prob_distribution).item()
    next_token = index_to_rel[next_token_index]
    relation_sequence = (relation_sequence[0] + [next_token], )
    if next_token == END_TOKEN:
      break
  return relation_sequence[0]