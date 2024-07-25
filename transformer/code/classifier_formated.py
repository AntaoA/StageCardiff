import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
from module_transformer import END_TOKEN, SEP_TOKEN, PAD_TOKEN, TextGen, TextDataset
from transformer_train import train
from transformer_param import device, END_TOKEN, SEP_TOKEN, chemin_t, chemin_t_data, chemin_data_train, chemin_data_validation
from transformer_param import SEQUENCE_LENGTH, BATCH_SIZE, epochs, learning_rate, embed_dim, num_layers, num_heads
from math import exp, log
import random


seuil = 0.2

path = []

with open(chemin_t_data + "classification_path.txt", 'r') as f:
    for line in f:
        path += [line.split(" : ")]



def formating_path(path, rel, prob):
    path = path.split(" ")
    r1 = random.randint(0, 100000)
    r2 = random.randint(0, 100000)
    
    res = str(r1) + "\t" + str(r2) + "\t" + prob + "\t" + rel + "(X,Y) <= "
    last = "X"
    for i in range(1, len(path)-2):
        r = path[i]
        r = r[1:-2]
        dep = last
        arr = chr(i + 64)
        last = arr
        if r[-2:] == "-1":
            r = r[:-2]
            temp = dep
            dep = arr
            arr = temp
        res += r + "(" + dep + "," + arr + "),"
    r = path[-2]
    r = r[1:-2]
    dep = last
    arr = "Y"
    if r[-2:] == "-1":
        r = r[:-2]
        temp = dep
        dep = arr
        arr = temp
    res += r + "("+ dep + "," + arr + ")\n"
    return res



with open(chemin_t_data + "classification_path_formated.txt", 'w') as f:
    path_valide = []
    for l in path:
        p, rel_tgt, rel_src, prob_class, prob_path = l
        prob = float(prob_class)
        prob_path = float(prob_path)
        if rel_tgt == rel_src:
            if prob > seuil:
                f.write(formating_path(p, rel_tgt, str(prob_path)))