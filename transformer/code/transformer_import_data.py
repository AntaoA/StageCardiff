import numpy as np
import networkx as nx
import random
import os
import pickle
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN
from transformer_param import chemin_data, nb_paths, chemin_t_data

# Dataset Preparation
if os.path.exists(chemin_data + 'index.pickle'):
    with open(chemin_data + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab = pickle.load(f)
else:
    rel_vocab = []

    rel_vocab.append(START_TOKEN)
    rel_vocab.append(END_TOKEN)
    rel_vocab.append(PAD_TOKEN)
    rel_vocab.append(SEP_TOKEN)

    lines = []

    with open(chemin_data + 'list_rel.txt', 'r') as file:
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

    with open(chemin_data + 'index.pickle', 'wb') as f:
        pickle.dump((int_to_rel, rel_to_int, rel_vocab), f)

if os.path.exists(chemin_data + 'graphe_train.pickle'):
    with open(chemin_data + 'graphe_train.pickle', 'rb') as f:
        G = pickle.load(f)
else:
    G = nx.MultiGraph()
    # Ajout des noeuds et des arêtes au graphe    
    with open(chemin_data + 'train.txt', 'r') as file:
        for line in file:
            n1, r, n2 = line.strip().split('\t')
            G.add_node(n1)
            G.add_node(n2)
            G.add_edge(n1, n2, relation = r)
            G.add_edge(n2, n1, relation = r + '-1')
    with open(chemin_data + 'graphe_train.pickle', 'wb') as f:
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

if os.path.exists(chemin_data + 'list_path.pickle'):
    with open(chemin_data + 'list_path.pickle', 'rb') as f:
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
    
    with open(chemin_data + 'list_path.pickle', 'wb') as f:
        pickle.dump(samples, f)


rel_vocab_size = len(rel_vocab)

with open (chemin_t_data + "stats_relations.txt", "w") as f:
    for r in rel_vocab:
        if r[-6:] == "_input":
            count = 0
            count = sum(1 for i in samples if r == i[0])
            f.write(f"{r} : {count}\n")
