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

chemin = "FB15K-237/Data/"



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


# marche aléatoire qui prend deux noeuds, une relation, et qui interdit d'utiliser cette relation directement
def random_walk(graph, start_node, end_node, relation, max_length, alpha=1.0, rti=rel_to_index, S=START_TOKEN, P=PAD_TOKEN, E=END_TOKEN):
    path = [rti[S]]
    current_node = start_node
    trouve = False
    for i in range(max_length-2):
        neighbors = list(graph.successors(current_node))

        if not neighbors:
            break
        
        # Calculer les probabilités pour choisir le prochain nœud
        distances = np.array([nx.shortest_path_length(graph, node, end_node) if nx.has_path(graph, node, end_node) else np.inf for node in neighbors])

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
        path.append(rti[r])
        current_node = next_node
        if current_node == end_node:
            trouve = True
            path.append(rti[E])
            path += [rti[P]] * (max_length - i - 3)
            break
    if not trouve:
        path = []
    return path



def random_walk_networkx(graph, start_node, end_node, relation, max_length, rti=rel_to_index, S=START_TOKEN, P=PAD_TOKEN, E=END_TOKEN):
    current_node = start_node
    trouve = False
    generator = nx.all_simple_paths(G, start_node, end_node, max_length-2)
    list_path = list(generator)
    path = random.choice(list_path)
    for i in range(1,len(path-1)):
        next_node = path[i]
        d = graph.get_edge_data(current_node, next_node)
        r = d[random.randint(0, (len(d)-1))]['relation']
        if i == 0:
            if r == relation:
                break
        path.append(rti[r])
        current_node = next_node
        if current_node == end_node:
            trouve = True
            path.append(rti[E])
            path += [rti[P]] * (max_length - i - 3)
            break
    if not trouve:
        path = []
    return path



NEG_INFTY = -1e9


# paramètres
d_model = 512
batch_size = 10
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.1
num_layers = 1
src_length = 1
tgt_length = 6
nb_paths = 1000



# création des données de train

rel_src = []
rel_tgt = []

i = 0
while i < nb_paths:
    edge = random.choice(list(G.edges(data=True)))
    node1, node2, r = edge[0], edge[1], edge[2]['relation']
    path = random_walk_networkx(G, node1, node2, r, tgt_length)
    if not path == []:
        print(i)
        i = i+1
        rel_src.append([rel_to_index[r]])
        rel_tgt.append(path)
    
transformer = t.Transformer(d_model, 
                            ffn_hidden,
                            num_heads, 
                            drop_prob, 
                            num_layers, 
                            tgt_length,
                            rel_vocab_size,
                            rel_to_index,
                            START_TOKEN, 
                            END_TOKEN,
                            PAD_TOKEN)




class TextDataset(t.Dataset):
    
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
    
dataset = TextDataset(rel_src, rel_tgt)

train_loader = t.DataLoader(dataset, batch_size, collate_fn = dataset.collate_fn)
iterator = iter(train_loader)



criterian = t.nn.CrossEntropyLoss(ignore_index=rel_to_index[PAD_TOKEN],
                                reduction='none')


# When computing the loss, we are ignoring cases when the label is the PAD token
for params in transformer.parameters():
    if params.dim() > 1:
        t.nn.init.xavier_uniform_(params)

optim = t.torch.optim.Adam(transformer.parameters(), lr=1e-4)
device = t.torch.device('cuda')



transformer.train()
transformer.to(device)
total_loss = 0
num_epochs = 10


for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    iterator = iter(train_loader)
    for batch_num, batch in enumerate(iterator):
        transformer.train()
        src, tgt = batch
        src = torch.tensor(src).to(device)
        tgt = torch.tensor(tgt).to(device)
        src_self_attention_mask, tgt_self_attention_mask, cross_attention_mask = t.create_masks(src, tgt, src_length, tgt_length, rel_to_index[PAD_TOKEN], NEG_INFTY)
        optim.zero_grad()
        rel_predictions = transformer(  src,
                                        tgt,
                                        src_self_attention_mask.to(device), 
                                        tgt_self_attention_mask.to(device), 
                                        cross_attention_mask.to(device))
        loss = criterian(
            rel_predictions.view(-1, rel_vocab_size).to(device),
            tgt.view(-1).to(device)
        ).to(device)
        valid_indicies = t.torch.where(tgt.view(-1) == rel_to_index[PAD_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"Input: {src[0]}")
            print(f"Output: {tgt[0]}")
            rel_sentence_predicted = t.torch.argmax(rel_predictions[0], axis=1)
            predicted_sentence = []
            for idx in rel_sentence_predicted:
              if idx == rel_to_index[END_TOKEN]:
                break
              predicted_sentence += [idx.item()]
            print(f"Path: {predicted_sentence}")
            print("-------------------------------------------")


transformer.eval()
def predict(num_rel):
  src = torch.full((1, src_length), num_rel).to(device)
  tgt = torch.full((1, tgt_length), rel_to_index[PAD_TOKEN]).to(device)
  for i in range(tgt_length):
    src_self_attention_mask, tgt_self_attention_mask, cross_attention_mask= t.create_masks(src, tgt,  src_length, tgt_length, rel_to_index[PAD_TOKEN], NEG_INFTY)
    predictions = transformer(src,
                              tgt,
                              src_self_attention_mask.to(device), 
                              tgt_self_attention_mask.to(device), 
                              cross_attention_mask.to(device))
    next_token_prob_distribution = predictions[0][i]
    next_token = t.torch.argmax(next_token_prob_distribution).item()
    tgt[0][i] = next_token
    if next_token == rel_to_index[END_TOKEN]:
      break
  return tgt[0]

print("Résultat de la prédiction")
print(predict(24))