import numpy as np
import networkx as nx
import random
import os
import pickle
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH
from transformer_param import chemin_t_data, nb_paths_per_relation
from transformer_param import chemin_data_train as chemin_data
import transformer_param as tp


with open(chemin_data + 'list_path.pickle', 'rb') as f:
    s, r, l = pickle.load(f)
    

samples = []
for p in s:
    samples.append(p + [PAD_TOKEN] * (SEQUENCE_LENGTH - len(p)))

with open(chemin_data + 'list_path.pickle', 'wb') as f:
    pickle.dump([samples, r, l], f)