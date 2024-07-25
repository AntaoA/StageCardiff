import numpy as np
import networkx as nx
import random
import os
import pickle
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH
from transformer_param import chemin_t_data, nb_paths_per_triplet
from transformer_param import chemin_data, chemin_data_train     as chemin
import transformer_param as tp

#chemin_data = "grail-master/data/fb237_v4_ind/"

# Dataset Preparation
if os.path.exists(chemin + 'index.pickle'):
    with open(chemin + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    print("Error: missing data")

def inv(relation):
    if relation[-2:] == '-1':
        return relation[:-2]
    else:
        return relation + '-1'


def inv_path(path):
    inv_path = []
    path_rev = path
    path_rev.reverse()
    for r in path_rev:
        inv_path.append(inv(r))
    return inv_path

if os.path.exists(chemin + 'graphe_train.pickle'):
    with open(chemin + 'graphe_train.pickle', 'rb') as f:
        G, triplet_from_rel = pickle.load(f)
else:
    print("Error: missing data")



def random_walk(start_node, end_node, relation, max_length, graph=G, alpha=1.0, rti=rel_to_int, S=START_TOKEN, P=PAD_TOKEN, E=END_TOKEN):
    path = [rti[S]]
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





if os.path.exists(chemin + 'list_6_path.pickle'):
    with open(chemin_data + 'list_6_path.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)
else:
    
    rel_src = []
    rel_tgt = []
    
    if chemin[-3::] == "in/":
        for j in range(len(vocab_input)):
            print(f"Relation {j}")
            if j % 2 == 0:
                for i,(s_node, e_node) in enumerate(triplet_from_rel[j]):
                    print(f"Triplet {i} sur {len(triplet_from_rel[j])}")
                    for k in range(nb_paths_per_triplet):
                        p = random_walk(s_node, e_node, j, 8, G)
                        while p == []:
                            p = random_walk(s_node, e_node, j, 8, G)
                        rel_src.append(int_to_rel_input[j])
                        p_s = []
                        for i in range(len(p)):
                            p_s.append(int_to_rel[p[i]])
                        rel_tgt.append(' '.join(p_s))
                        
                        p = random_walk(s_node, e_node, j, 8, G)
                        while p == []:
                            p = random_walk(s_node, e_node, j, 8, G)
                        p_s = []
                        for i in range(len(p)):
                            p_s.append(int_to_rel[p[i]])
                        rel_src.append(int_to_rel_input[j+1])
                        rel_tgt.append(' '.join(inv_path(p_s)))
                    
        samples = [[r1, SEP_TOKEN, START_TOKEN] + r2.split(' ') + [END_TOKEN] + [PAD_TOKEN] * (SEQUENCE_LENGTH - 4 - len(r2.split(' '))) for r1, r2 in zip(rel_src, rel_tgt)]

        with open(chemin + 'list_6_path.pickle', 'wb') as f:
            pickle.dump((samples, rel_src, rel_tgt), f)
    else:
        for i in range(len(vocab_input)):
            if i % 2 == 0:
                print(f"Relation {i} - Nombre triplet: {len(triplet_from_rel[i])}")
                for n1, n2 in triplet_from_rel[i]:
                    
                    paths = list(nx.all_simple_paths(G, source=n1, target=n2, cutoff=6))
                    paths_valid = []
                    last_rel = (int_to_rel_input[i])[:-6]
                    for path in paths:
                        paths_rel = [[]]
                        for j in range(len(path) - 1):
                            d = G.get_edge_data(path[j], path[j+1])
                            new_path_rel = []
                            for p in paths_rel:
                                last_rel = p[-1] if len(p)>0 else last_rel
                                for r in d:
                                    new_p = p + [d[r]['relation']]
                                    if d[r]['relation'] == last_rel and path[j] == n1 and path[j+1] == n2:
                                        continue
                                    if j < 1 or path[j-1] != path[j+1] or inv(d[r]['relation'] != last_rel):
                                        new_path_rel.append(new_p)
                            paths_rel = new_path_rel
                        paths_valid += paths_rel
                    
                    for p in paths_valid:
                        rel_src.append(int_to_rel_input[i])
                        rel_tgt.append(' '.join(p))
                        
                        rel_src.append(int_to_rel_input[i+1])
                        rel_tgt.append(' '.join(inv_path(p)))
    
        samples = [[r1, SEP_TOKEN, START_TOKEN] + r2.split(' ') + [END_TOKEN] + [PAD_TOKEN] * (SEQUENCE_LENGTH - 4 - len(r2.split(' '))) for r1, r2 in zip(rel_src, rel_tgt)]
        
        with open(chemin + 'list_path.pickle', 'wb') as f:
            pickle.dump((samples, rel_src, rel_tgt), f)
    


rel_vocab_size = len(rel_vocab)

with open (chemin_t_data + "stats_relations.txt", "w") as f:
    for r in rel_vocab:
        if r[-6:] == "_input":
            count = 0
            count = sum(1 for i in samples if r == i[0])
            f.write(f"{r} : {count}\n")