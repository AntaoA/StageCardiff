import random
import os
import pickle
import networkx as nx
from transformer_param import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH, BATCH_SIZE, device
from transformer_param import chemin_t_data, nb_paths_per_triplet_fine_tuning, name_transformer
from transformer_param import chemin_data, chemin_t, chemin_data_validation, chemin_data_test, chemin_data_train as chemin
from transformer_param import epochs, learning_rate
import torch.nn as nn
from torch.utils.data import DataLoader
from module_transformer import TextDataset
from torch.utils.data import DataLoader
import os
import pickle
from transformer_train import train
import torch.optim as optim
import torch
from math import log, exp
import torch.nn.functional as F
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

if os.path.exists(chemin_data_validation + 'list_path.pickle'):
    with open(chemin_data_validation + 'list_path.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)
else:
    print("Erreur: fichier list_path.pickle non trouvé")


def create_path(i):
    rel_tgt = []
    inverse = False
    if i % 2 == 1:
        i = i-1
        inverse = True
    for t in range(len(triplet_from_rel[i])):
        with open(chemin + "list_paths/" + "rel_"+str(i) + '/triplet_' + str(t) + '.pickle', 'rb') as g:
            list_paths = pickle.load(g)
            for _ in range(nb_paths_per_triplet_fine_tuning):
                if list_paths == []:
                    break
                if inverse:
                    p = random.choice(list_paths)
                    rel_tgt.append(' '.join(inv_path(p)))
 
                else:
                    p = random.choice(list_paths)
                    rel_tgt.append(' '.join(p))
                
    samples = [[int_to_rel_input[i], SEP_TOKEN, START_TOKEN] + r.split(' ') + [END_TOKEN] + [PAD_TOKEN] * (SEQUENCE_LENGTH - 4 - len(r.split(' '))) for r in rel_tgt]
    return samples


with open(chemin_t_data + 'stats_model_fine_tuning.txt', 'w') as h:
    for i in range( len(triplet_from_rel)):
        if os.path.exists(chemin_t + '/fine_tuning/fine_tuned_model_' + str(i) + '.pickle'):
            print(f"Rel {i} already fine tuned")
            continue
        samples = create_path(i)
        with open (chemin_t + name_transformer, 'rb') as f:
            model = pickle.load(f)
        open(chemin_t + '/fine_tuning/fine_tuned_model_' + str(i) + '.pickle', 'wb').write(pickle.dumps(model))
        dataset = TextDataset(samples, rel_to_int)
        dataloader = DataLoader(
            dataset, 
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        #print(f"Triplet {i} - Batch {len(dataloader)}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)



        def calculate_perplexity(model, rel_src=rel_src, rel_tgt=rel_tgt):
            model.eval()
            model.to(device)

            sum_prob = 0.0
            total_tokens = 0

            indices = []
            for k in range(len(rel_src)):
                if rel_src[k] == int_to_rel_input[i]:
                    indices.append(k)
                    
            # Sélectionnez les éléments correspondants dans les deux listes
            src = [rel_src[i] for i in indices]
            tgt = [rel_tgt[i] for i in indices]

            if len(src) == 0:
                return 0, 0
            
            
            with torch.no_grad():
                
                for k,path_txt in enumerate(tgt):
                    path = [START_TOKEN] + path_txt.split() + [END_TOKEN]
                    int_list = [rel_to_int[src[k]], rel_to_int[SEP_TOKEN]]
                    path_sum = 0
                    path_tok = 0
                    
                    for j in range(len(path)-1):
                        model.eval()
                        int_list.append(rel_to_int[path[j]])
                        int_vector = torch.tensor(int_list).unsqueeze(0).to(device)

                        predictions = model(int_vector)
                        
                        prob = F.softmax(predictions[:, -1, :], dim=-1)[0][rel_to_int[path[j+1]]].item()
                        
                        path_sum += log(prob)
                        path_tok += 1
                        total_tokens += 1
                        sum_prob += log(prob)
            return exp(-sum_prob / total_tokens), total_tokens

    
        model_copy = model
        initial_perplexity, _ = calculate_perplexity(model)
        #fine tuning
        model_copy, best_perplexity = train(model_copy, 10, dataloader, criterion, optimizer, calculate_perplexity)
        print(f"Rel {i} - Initial Perplexity: {initial_perplexity} -- Best Perplexity: {best_perplexity}")
        h.write(f"Rel {i} - Initial Perplexity: {initial_perplexity} -- Best Perplexity: {best_perplexity}\n")
        if best_perplexity < initial_perplexity:
            print("Fine tuning successful")
            open(chemin_t + '/fine_tuning/fine_tuned_model_' + str(i) + '.pickle', 'wb').write(pickle.dumps(model_copy))
        else:
            print("Fine tuning failed")
if False:
    sum_prob = 0.0
    sum_prob_ft = 0.0
    total_tokens = 0

    src = rel_src
    tgt = rel_tgt

    with open(chemin_t_data + 'stats_model_model_ft.txt', 'w') as f:

        with torch.no_grad():
            
            tab_tok = [0] * len(rel_to_int_input)
            tab_prob = [0.0] * len(rel_to_int_input)
            tab_prob_ft = [0.0] * len(rel_to_int_input)
            
            for k,path_txt in enumerate(tgt):
                path = [START_TOKEN] + path_txt.split() + [END_TOKEN]
                int_list = [rel_to_int[src[k]], rel_to_int[SEP_TOKEN]]
                path_sum = 0
                path_sum_ft = 0
                path_tok = 0
                
                
                with open(chemin_t + '/fine_tuning/fine_tuned_model_' + str(rel_to_int_input[src[k]]) + '.pickle', 'rb') as g:
                    with open (chemin_t + name_transformer, 'rb') as h:
                        model = pickle.load(h)
                    model_ft = pickle.load(g)
                    for j in range(len(path)-1):
                        model_ft.eval()
                        model_ft.to(device)
                        model.eval()
                        model.to(device)
                        
                        int_list.append(rel_to_int[path[j]])
                        int_vector = torch.tensor(int_list).unsqueeze(0).to(device)

                        predictions = model(int_vector)
                        predictions_ft = model_ft(int_vector)
                        
                        prob = F.softmax(predictions[:, -1, :], dim=-1)[0][rel_to_int[path[j+1]]].item()
                        prob_ft = F.softmax(predictions_ft[:, -1, :], dim=-1)[0][rel_to_int[path[j+1]]].item()
                        
                        path_sum += log(prob)
                        path_sum_ft += log(prob_ft)
                        path_tok += 1
                        total_tokens += 1
                        sum_prob += log(prob)
                        sum_prob_ft += log(prob_ft)
                        
                        
                        tab_tok[rel_to_int_input[src[k]]] += 1
                        tab_prob[rel_to_int_input[src[k]]] += log(prob)
                        tab_prob_ft[rel_to_int_input[src[k]]] += log(prob_ft)
                    
                    if k % 25 == 0:    
                        print(f"Path {k} -- Sum: {path_sum} -- Tokens: {path_tok} -- Perp: {exp(-path_sum / path_tok)}")
                        print(f"Path {k} -- Sum: {path_sum_ft} -- Tokens: {path_tok} -- Perp: {exp(-path_sum_ft / path_tok)}\n")
                    
                    f.write(f"Path {k} -- Sum: {path_sum} -- Tokens: {path_tok} -- Perp: {exp(-path_sum / path_tok)}\n")    
                    f.write(f"Path {k} -- Sum: {path_sum_ft} -- Tokens: {path_tok} -- Perp: {exp(-path_sum_ft / path_tok)}\n\n")
                    
                    
        print(exp(-sum_prob / total_tokens))
        print(exp(-sum_prob_ft / total_tokens))
        print(total_tokens)
        
        
    for i in range(len(tab_tok)):
        if tab_tok[i] != 0:
            print(f"Rel {i} -- Tokens {tab_tok[i]} -- Perp {exp(-tab_prob[i] / tab_tok[i])} -- Perp_ft {exp(-tab_prob_ft[i] / tab_tok[i])}")
        else:
            print(f"Rel {i} -- Tokens {tab_tok[i]} -- Perp 0 -- Perp_ft 0")
            
    with open(chemin_t_data + 'tableau_model.pickle', 'wb') as f:
        pickle.dump([tab_tok, tab_prob, tab_prob_ft], f)
    

if False:
    if os.path.exists(chemin_data_test + 'list_path.pickle'):
        with open(chemin_data_test + 'list_path.pickle', 'rb') as f:
            samples, rel_src, rel_tgt = pickle.load(f)
    else:
        print("Erreur: fichier list_path.pickle non trouvé")


    if os.path.exists(chemin_data_test + 'index.pickle'):
        with open(chemin_data_test + 'index.pickle', 'rb') as f:
            int_to_rel_t, rel_to_int_t, rel_vocab_t, vocab_input_t, rel_to_int_input_t, int_to_rel_input_t = pickle.load(f)
    else:
        print("Erreur: fichier index.pickle non trouvé")



    sum_prob = 0.0
    total_tokens = 0

        
    with torch.no_grad():
        for i,path_txt in enumerate(rel_tgt):
            path = [START_TOKEN] + path_txt.split() + [END_TOKEN]
            int_list = [rel_to_int[rel_src[i]], rel_to_int[SEP_TOKEN]]
            path_sum = 0
            path_tok = 0
            with open(chemin_t + '/fine_tuning/fine_tuned_model_' + str(rel_to_int_input[rel_src[i]]) + '.pickle', 'rb') as f:
                model = pickle.load(f)
            for j in range(len(path)-1):
                model.eval()
                model.to(device)

                int_list.append(rel_to_int[path[j]])
                int_vector = torch.tensor(int_list).unsqueeze(0).to(device)

                predictions = model(int_vector)

                prob = F.softmax(predictions[:, -1, :], dim=-1)[0][rel_to_int[path[j+1]]].item()

                path_sum += log(prob)
                path_tok += 1
                total_tokens += 1
                sum_prob += log(prob)
            print(f"Path {i} -- Sum: {path_sum} -- Tokens: {path_tok} -- Perp: {exp(-path_sum / path_tok)}")
    print(exp(-sum_prob / total_tokens))
    print(total_tokens)
