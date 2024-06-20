import module_transformer as t
import numpy as np
import random
import networkx as nx


    
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

transformer = t.Transformer(d_model, 
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
    
dataset = TextDataset(nodes_input, relation_sequences)

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
        nodes_batch, rel_batch = batch
        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = t.create_masks(nodes_batch, rel_batch, max_sequence_length, NEG_INFTY)
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
        valid_indicies = t.torch.where(labels.view(-1) == rel_to_index[PAD_TOKEN], False, True)
        loss = loss.sum() / valid_indicies.sum()
        loss.backward()
        optim.step()
        #train_losses.append(loss.item())
        if batch_num % 100 == 0:
            print(f"Iteration {batch_num} : {loss.item()}")
            print(f"Nodes: {nodes_batch[0]}")
            print(f"Path: {rel_batch[0]}")
            rel_sentence_predicted = t.torch.argmax(rel_predictions[0], axis=1)
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
    encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask= t.create_masks(nodes, relation_sequence, max_sequence_length, NEG_INFTY)
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
    next_token_index = t.torch.argmax(next_token_prob_distribution).item()
    next_token = index_to_rel[next_token_index]
    relation_sequence = (relation_sequence[0] + [next_token], )
    if next_token == END_TOKEN:
      break
  return relation_sequence[0]