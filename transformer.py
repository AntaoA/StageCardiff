import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import random
import networkx as nx
import pickle
import numpy as np
import module_transformer as t
from module_transformer import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH
from module_transformer import TextDataset, TextGen

chemin = "grail-master/data/fb237_v4/"

nb_paths = 1000

# Dataset Preparation

if os.path.exists(chemin + 'index.pickle'):
    with open(chemin + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab = pickle.load(f)
else:
    rel_vocab = []

    rel_vocab.append(START_TOKEN)
    rel_vocab.append(END_TOKEN)
    rel_vocab.append(PAD_TOKEN)
    rel_vocab.append(SEP_TOKEN)

    lines = []

    with open(chemin + 'list_rel.txt', 'r') as file:
        lines += file.readlines()    
    


    # Ajouter les relation et les relations inverses
    for line in lines:
        line = line.strip()
        rel_vocab.append(line)
        rel_vocab.append(line + '_input')
        rel_vocab.append(line + '-1')
        rel_vocab.append(line + '-1_input')

    int_to_rel = {k:v.strip() for k,v in enumerate(rel_vocab)}
    rel_to_int = {v.strip():k for k,v in enumerate(rel_vocab)}

    with open(chemin + 'index.pickle', 'wb') as f:
        pickle.dump((int_to_rel, rel_to_int, rel_vocab), f)

if os.path.exists(chemin + 'graphe_train.pickle'):
    with open(chemin + 'graphe_train.pickle', 'rb') as f:
        G = pickle.load(f)
else:
    G = nx.MultiGraph()
    # Ajout des noeuds et des arêtes au graphe    
    with open(chemin + 'train.txt', 'r') as file:
        for line in file:
            n1, r, n2 = line.strip().split('\t')
            G.add_node(n1)
            G.add_node(n2)
            G.add_edge(n1, n2, relation = r)
            G.add_edge(n2, n1, relation = r + '-1')
    with open(chemin + 'graphe_train.pickle', 'wb') as f:
        pickle.dump(G, f)

def inv(relation):
    if relation[-2:] == '-1':
        return relation[:-2]
    else:
        return relation + '-1'


def random_walk(start_node, end_node, relation, max_length, graph, alpha, S=START_TOKEN, P=PAD_TOKEN, E=END_TOKEN):
    path = [S]
    current_node = start_node
    trouve = False
    last_r = relation
    last_n = end_node
    for i in range(max_length-2):
        neighbors = list(graph.neighbors(current_node))
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
        
        if r == inv(last_r) and next_node == last_n:
            break
        last_r = r
        last_n = current_node
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

if os.path.exists(chemin + 'list_path.pickle'):
    with open(chemin + 'list_path.pickle', 'rb') as f:
        samples = pickle.load(f)
else:
    rel_src = []
    rel_tgt = []
    
    i = 0
    
    while i < nb_paths:
        edge = random.choice(list(G.edges(data=True)))
        node1, node2, r = edge[0], edge[1], edge[2]['relation']
        path = random_walk(node1, node2, r, 6, G, 2)
        if not path == []:
            print(i)
            i = i+1
            rel_src.append(r + '_input')
            rel_tgt.append(' '.join(path))
        
        
    samples = [[r1, SEP_TOKEN] + r2.split(' ') for r1, r2 in zip(rel_src, rel_tgt)]
    
    with open(chemin + 'list_path.pickle', 'wb') as f:
        pickle.dump(samples, f)


rel_vocab_size = len(rel_vocab)

with open ("quelques_relations.txt", "w") as f:
    for r in rel_vocab:
        if r[-6:] == "_input":
            count = 0
            count = sum(1 for i in samples if r == i[0])
            f.write(f"{r}\n{count}\n\n")

BATCH_SIZE = 32

dataset = TextDataset(samples, rel_to_int)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training
def train(model, epochs, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for input_seq, target_seq, padding_mask in dataloader:
            input_seq, target_seq, padding_mask = input_seq.to(device), target_seq.to(device), padding_mask.to(device)
            
            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, rel_vocab_size)
            
            active_loss = padding_mask.view(-1) == 1
            active_logits = outputs.view(-1, rel_vocab_size)[active_loss]
            active_labels = target_seq.view(-1)[active_loss]
            
            loss = criterion(active_logits, active_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().numpy()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} loss: {epoch_loss:.3f}")

epochs = 100
learning_rate = 0.001 
embed_dim=100
num_layers=2 
num_heads=2



if os.path.exists('transformer.pickle'):
    with open('transformer.pickle', 'rb') as f:
        model = pickle.load(f)
else:
    model = TextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=100,
        num_layers=2, 
        num_heads=2,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    train(model, epochs, dataloader, criterion, optimizer) 
    open('transformer.pickle', 'wb').write(pickle.dumps(model))

def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([rel_to_int[word] for word in words[-t.SEQUENCE_LENGTH:]]).unsqueeze(0)
    return input_seq

def sample_next(predictions, k):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    topk_probs, topk_indices = torch.topk(probabilities, k)
    return topk_probs[0], topk_indices[0]

def text_generator(sentence, generate_length):
    model.eval()
    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample)
        if len(int_vector) >= SEQUENCE_LENGTH - 1:
            break
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions, 1)
        sample += ' ' + int_to_rel[next_token]
        if next_token == rel_to_int[END_TOKEN]:
            break
    print(sample)
    print('\n')
    
    
def text_generator_with_confidence(sentence, generate_length, k):
    model.eval()
    samples = [(" ".join(sentence.split()), 1.0, [])]        
    for _ in range(generate_length):
        all_candidates = []
        for seq, score, list_prob in samples:
            if seq[-5:] == END_TOKEN:
                all_candidates.append((seq, score, list_prob))
                continue
            int_vector = return_int_vector(seq).to(device)

            if len(int_vector) >= SEQUENCE_LENGTH - 1:
                break

            with torch.no_grad():
                predictions = model(int_vector)
                
            
            topk_probs, topk_indices = sample_next(predictions, k)
            
            
            for i in range(k):
                next_token_index = topk_indices[i].item()
                next_token_prob = topk_probs[i].item()
                next_token = int_to_rel[next_token_index]
                candidate = (seq + " " + next_token, score * next_token_prob, list_prob + [next_token_prob])                                
                all_candidates.append(candidate)
            
        # Order all candidates by their probability scores
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        # Select the k best candidates
        samples = ordered[:k]    
        
    return samples

    
sentence = "/award/award_nominee/award_nominations./award/award_nomination/award_nominee-1_input <SEP> <START> /award/award_nominee/award_nominations./award/award_nomination/award_nominee"

paths = [i[3:-1] for i in samples if i[0] == "/award/award_nominee/award_nominations./award/award_nomination/award_nominee-1_input"]

generate_length = 8

print(f"\n\nPROMPT: {sentence}\n")    
out = text_generator_with_confidence(sentence, generate_length, 5)

for i in range(len(out)):
    r, p, pl = out[i]
    for t in r.split():
        print(t)
    print(f"{p} - {pl} \n")
    

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK




# Espace de recherche des hyperparamètres
space = {
    'epochs': hp.quniform('epochs', 10, 100, 10),
    'batch_size': hp.quniform('batch_size', 8, 128, 8),
    'num_layers': hp.quniform('num_layers', 1, 8, 1),
    'num_heads': hp.quniform('num_heads', 1, 8, 1),
    'embed_dim': hp.quniform('embed_dim', 10, 100, 10),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01)
}
def objective(params):

    # Afficher les paramètres de l'essai en cours
    print("Testing parameters:", params)
    

    # Charger les données
    dataset = TextDataset(samples, rel_to_int)
    dataloader = DataLoader(dataset, batch_size=int(params['batch_size']), shuffle=True)    
    
    # Définir le modèle
    model = TextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=int(params['embed_dim']*params['num_heads']),
        num_layers=int(params['num_layers']), 
        num_heads=int(params['num_heads'])
    ).to(device)
    
    # Définir le critère et l'optimisateur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    # Entraînement
    train(model, int(params['epochs']), dataloader, criterion, optimizer)
    # Calculer la perte (ou une autre métrique) pour évaluer le modèle
    # Utiliser une simple moyenne de la perte d'entraînement pour cet exemple
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for input_seq, target_seq, padding_mask in dataloader:
            input_seq, target_seq, padding_mask = input_seq.to(device), target_seq.to(device), padding_mask.to(device)
            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, rel_vocab_size)
            active_loss = padding_mask.view(-1) == 1
            active_logits = outputs.view(-1, rel_vocab_size)[active_loss]
            active_labels = target_seq.view(-1)[active_loss]
            loss = criterion(active_logits, active_labels)
            running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    
    # Minimiser la perte moyenne
    return {'loss': avg_loss, 'status': STATUS_OK}

# Définir les essais pour stocker les résultats
trials = Trials()

# Exécuter l'optimisation
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)

print("Best hyperparameters found:", best)