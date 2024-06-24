import module_transformer as m
import transformer as t
import os
import pickle
import networkx as nx
import random
import numpy as np 
import torch

chemin = "grail-master/data/fb237_v4_ind/"
#chemin = "FB15K-237/Data/"



START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'


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