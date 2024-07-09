import numpy as np
import networkx as nx
import random
import os
import pickle
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH
from transformer_param import chemin_t_data, nb_paths_per_triplet
from transformer_param import chemin_data, chemin_data_train as chemin
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


if os.path.exists(chemin + 'list_path.pickle'):
    with open(chemin_data + 'list_path.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)
else:
    
    rel_src = []
    rel_tgt = []
    
    if chemin[-3::] == "in/":
        for j in range(len(vocab_input)):
            print(f"Relation {j}")
            if j % 2 == 0:
                list_paths = []     
                i = 0
                while os.path.exists(chemin + "list_paths/" + "rel_"+str(j) + '/triplet_' + str(i) + '.pickle'):
                    with open(chemin + "list_paths/" + "rel_"+str(j) + '/triplet_' + str(i) + '.pickle', 'rb') as g:
                        list_paths = pickle.load(g)
                    for k in range(nb_paths_per_triplet):
                        if list_paths == []:
                            break
                        p = random.choice(list_paths)
                        rel_src.append(int_to_rel_input[j])
                        rel_tgt.append(' '.join(p))
                        
                        p = random.choice(list_paths)
                        rel_src.append(int_to_rel_input[j+1])
                        rel_tgt.append(' '.join(inv_path(p)))
                    i += 1
                    
        samples = [[r1, SEP_TOKEN, START_TOKEN] + r2.split(' ') + [END_TOKEN] + [PAD_TOKEN] * (SEQUENCE_LENGTH - 4 - len(r2.split(' '))) for r1, r2 in zip(rel_src, rel_tgt)]

        with open(chemin + 'list_path_10.pickle', 'wb') as f:
            pickle.dump((samples, rel_src, rel_tgt), f)
    else:
        for j in range(len(vocab_input)):
            print(f"Relation {j}")
            if j % 2 == 0:
                list_paths = []            
                with open(chemin + "list_paths/" + "rel_"+str(j) + '.pickle', 'rb') as g:
                    list_paths = pickle.load(g)
                for triplet in list_paths:
                    for p in triplet:
                        rel_src.append(int_to_rel_input[j])
                        rel_tgt.append(' '.join(p))
                        
                        rel_src.append(int_to_rel_input[j+1])
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