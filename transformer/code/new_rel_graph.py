import networkx as nx
import os
import pickle
import random
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH
from transformer_param import chemin_t_data, chemin_data_train, nb_paths_per_triplet
import ast


with open(chemin_t_data + "chemin_to_relation.txt", "r") as f:
    chemin_to_relation = f.read().split("\n")
    
with open(chemin_data_train + "graphe_train.pickle", "rb") as f:
    G, triplet_from_rel = pickle.load(f)
    

if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_train + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)


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


def rti_no_input(k):
    return rel_to_int[k]//2


rel_src = []
rel_tgt = []



for j in range(len(vocab_input)):                                                                                               
    print(f"Relation {j}")                                                                                              
    if j % 2 == 0:                                                                                                  
        list_paths = []                                                                                                 
        i = 0                                                                                                           
        for ctr in chemin_to_relation:                                                                      
            path, new_rel, prob = ctr.split(" : ")
            path = ast.literal_eval(path)
            potential_path = []
            start = True
            for rel in path[1:-1]:      # Pour gérer le <START> et le <END>
                for t in triplet_from_rel[rti_no_input(rel)]:
                    if start:
                        start = False
                        potential_path.append(t)
                    else:
                        for i,p in enumerate(potential_path):
                            if p[-1] == t[0]:
                                potential_path[i] = p + t[1:]
            for p in potential_path:
                head = p[0]
                tail = p[-1]
                triple = G.get_edge_data(head, tail)
            i += 1                                                                                                                  
