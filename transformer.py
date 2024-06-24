import module_transformer as t
import torch
import numpy as np
import random
import networkx as nx
import pickle
import os
    
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



transformer = t.Transformer(d_model, 
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
device = t.get_device()



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
                                dec_end_token=True)
        next_token_prob_distribution = predictions[0][word_counter]
        next_token_index = torch.argmax(next_token_prob_distribution).item()
        next_token = rel_to_index[next_token_index]
        tgt_sentence = (tgt_sentence[0] + next_token, )
        if next_token == END_TOKEN:
            break
    return tgt_sentence[0]