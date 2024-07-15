import networkx as nx
import os
import pickle
import random
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH
from transformer_param import chemin_t_data, chemin_data_train, nb_paths_per_triplet, chemin_data_test
import ast


with open(chemin_t_data + "chemin_to_relation.txt", "r") as f:
    chemin_to_relation = f.read().split("\n")
    
with open(chemin_data_test + "graphe_train.pickle", "rb") as f:
    G, triplet_from_rel = pickle.load(f)
    
    
new_G = G.copy()

if os.path.exists(chemin_data_test + 'index.pickle'):
    with open(chemin_data_test + 'index.pickle', 'rb') as f:
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
    if k[-3:] == '_of':
        print("feur")
    return rel_to_int_input[k + '_input']


rel_src = []
rel_tgt = []

sum_theorique = 0
sum = 0
list_paths = []                                                                                                 
for i,ctr in enumerate(chemin_to_relation[:-1]):                                                                   
    count_ajout = 0
    count_ajout_theorique = 0
    path, new_rel, prob = ctr.split(" : ")
    path = ast.literal_eval(path)
    potential_path = []
    start = True
    last_rel = inv(new_rel)
    for rel in path[1:-1]:      # Pour gérer le <START> et le <END>
        lp = []
        for t in triplet_from_rel[rti_no_input(rel)]:
            if start:
                lp.append(t)
            else:
                if potential_path == []:
                    break
                for p in potential_path:
                    if p[-1] == t[0] and not (p[-2] == t[-1] and rel == inv(last_rel)):
                        lp.append(p + t[1:])
        potential_path = lp
        if start:
            start = False
        last_rel = rel
    for p in potential_path:
        head = p[0]
        tail = p[-1]
        d = G.get_edge_data(head, tail)
        ajouter = True
        if G.has_edge(head, tail):
            for r in d:
                if d[r]['relation'] == new_rel:
                    ajouter = False
        if ajouter:    
            if random.random() < float(prob):
                new_G.add_edge(head, tail, relation=new_rel)
                count_ajout += 1                        
            count_ajout_theorique += 1
    sum_theorique += count_ajout_theorique
    sum += count_ajout
    print(f"Règle {i} : nombre_ajout {count_ajout} : nombre_ajout_theorique {count_ajout_theorique} : total_vrai {sum} : total_theorique {sum_theorique}")                                                                                                       
