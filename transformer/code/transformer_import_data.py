import numpy as np
import networkx as nx
import random
import os
import pickle
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN
from transformer_param import chemin_data, chemin_t_data, nb_paths_per_relation

# Dataset Preparation
if os.path.exists(chemin_data + 'index.pickle'):
    with open(chemin_data + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    rel_vocab = []

    rel_vocab.append(START_TOKEN)
    rel_vocab.append(END_TOKEN)
    rel_vocab.append(PAD_TOKEN)
    rel_vocab.append(SEP_TOKEN)

    lines = []

    vocab_input = []
    
    with open(chemin_data + 'list_rel.txt', 'r') as file:
        lines += file.readlines()    
    
    # Ajouter les relation et les relations inverses
    for line in lines:
        line = line.strip()
        rel_vocab.append(line)
        rel_vocab.append(line + '_input')
        rel_vocab.append(line + '-1')
        rel_vocab.append(line + '-1_input')

        vocab_input.append(line + '-1_input')
        vocab_input.append(line + '_input')

    int_to_rel = {k:v.strip() for k,v in enumerate(rel_vocab)}
    rel_to_int = {v.strip():k for k,v in enumerate(rel_vocab)}

    rel_to_int_input = {v.strip():k for k,v in enumerate(vocab_input)}
    int_to_rel_input = {k:v.strip() for k,v in enumerate(vocab_input)}

    with open(chemin_data + 'index.pickle', 'wb') as f:
        pickle.dump((int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input), f)

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

if os.path.exists(chemin_data + 'graphe_train.pickle'):
    with open(chemin_data + 'graphe_train.pickle', 'rb') as f:
        G, triplet_from_rel, liqt_path = pickle.load(f)
else:
    G = nx.MultiDiGraph()
    
    triplet_from_rel = [[] for _ in range(len(vocab_input))]
        
    # Ajout des noeuds et des arêtes au graphe    
    with open(chemin_data + 'train.txt', 'r') as file:
        for line in file:
            n1, r, n2 = line.strip().split('\t')
            G.add_node(n1)
            G.add_node(n2)
            G.add_edge(n1, n2, relation = r)
            G.add_edge(n2, n1, relation = r + '-1')
            
            triplet_from_rel[rel_to_int_input[r + '_input']] += [(n1, n2)]
            triplet_from_rel[rel_to_int_input[r + '-1_input']] += [(n2, n1)]


    somme_triplet = 0
    somme_path = 0
    with open(chemin_t_data + 'stats_globale_paths.txt', 'w') as f:
        for i in range(len(vocab_input)):
            name_file = chemin_data + "list_paths/" + "rel_"+str(i) + '.pickle'
            print(f"voc {i} : {len(triplet_from_rel[i])} triplets")
            if i % 2 == 0:
                list_paths = []
                m = 0
                for n1, n2 in triplet_from_rel[i]:
                    if m % 10 == 0:
                        print(m)
                    m += 1
                    paths = list(nx.all_simple_paths(G, n1, n2, cutoff=4))
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
                                    if d[r]['relation'] == last_rel and path[j] == n2 and path[j+1] == n1:
                                        continue
                                    if j < 1 or path[j-1] != path[j+1] or inv(d[r]['relation'] != last_rel):
                                        new_path_rel.append(new_p)
                            paths_rel = new_path_rel
                        paths_valid += paths_rel
                    list_paths += paths_valid
                somme_triplet += len(triplet_from_rel[i])
                somme_path += len(list_paths)
            
            
                with open(name_file, 'wb') as g:
                    pickle.dump(list_paths, g)
            
                f.write(f"voc {i} : {len(list_paths)} paths : {len(triplet_from_rel[i])} triplets")
                f.write(f"Jusqu'à présent : {somme_path} paths : {somme_triplet} triplets\n")
                print(f"{len(list_paths)} paths")
                print(f"Jusqu'à présent : {somme_path} paths : {somme_triplet} triplets\n")
            
    with open(chemin_data + 'graphe_train.pickle', 'wb') as f:
        pickle.dump((G, triplet_from_rel, list_paths), f)


if (r['relation'] != (int_to_rel_input[i])[:-6] or (path[j] != n1 or path[j+1] != n2)) and (j>=1 and (path[j-1] != [path[j+1]] or inv(r['relation']) != last_rel) ):
    True

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
    
    #for j in range(len(vocab_input)): 
    #    i = 0
    #    while i < nb_paths_per_relation:
    #        r = int_to_rel_input[j]
    #        node1, node2 = random.choice(triplet_from_rel[j])
    #        path = random_walk(node1, node2, r[:-6], 6, G, 2)
    #        if not path == []:
    #            print(j*nb_paths_per_relation + i)
    #            i = i+1
    #            rel_src.append(r)
    #            rel_tgt.append(' '.join(path))
        
    for j in range(len(vocab_input)):
        if j % 2 == 0:
            with open(chemin_data + "list_paths/" + "rel_"+str(j) + '.pickle', 'rb') as g:
                list_paths = pickle.load(g)
            for k in range (nb_paths_per_relation):
                path = random.choice(list_paths)
                rel_src.append(int_to_rel_input[j])
                rel_tgt.append(' '.join(path))
        else:
            with open(chemin_data + "list_paths/" + "rel_"+str(j-1) + '.pickle', 'rb') as g:
                list_paths = pickle.load(g)
            for k in range (nb_paths_per_relation):
                path = random.choice(list_paths)
                rel_src.append(int_to_rel_input[j])
                rel_tgt.append(' '.join(inv_path(path)))
    
    
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
