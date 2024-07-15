import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import os
import pickle
from transformer_param import chemin_t, device, chemin_data_train, name_transformer, chemin_data_validation, START_TOKEN, END_TOKEN, SEP_TOKEN
from transformer_param import SEQUENCE_LENGTH, BATCH_SIZE, epochs, learning_rate, embed_dim, num_layers, num_heads, n_gram, hidden_dim
import numpy as np
import copy
import random
from math import log, exp
from torch.nn import functional as F
from module_Ngram import NGramTextGen, TextDataset



name_Ngram = "ngram_model.pickle"

if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_train + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    print("Erreur : données manquantes")

rel_vocab_size = len(rel_vocab)

if os.path.exists(chemin_data_train + 'list_path_10.pickle'):
    with open(chemin_data_train + 'list_path_10.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)
else:
    print("Erreur : données manquantes")

dataset = TextDataset(samples, rel_to_int=rel_to_int, n=n_gram)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: x)


if os.path.exists(chemin_data_validation + 'list_path.pickle'):
    with open(chemin_data_validation + 'list_path.pickle', 'rb') as f:
        samples_v, rel_src_v, rel_tgt_v = pickle.load(f)
else:
    print("Erreur: fichier list_path.pickle non trouvé")


if os.path.exists(chemin_data_validation + 'index.pickle'):
    with open(chemin_data_validation + 'index.pickle', 'rb') as f:
        int_to_rel_v, rel_to_int_v, rel_vocab_v, vocab_input_v, rel_to_int_input_v, int_to_rel_input_v = pickle.load(f)
else:
    print("Erreur: fichier index.pickle non trouvé")




def calculate_perplexity(model):
    model.eval()
    model.to(device)

    sum_prob = 0.0
    total_tokens = 0

    # Obtenez 1000 indices aléatoires uniques   
    indices = random.sample(range(len(rel_src_v)), 1000)

    # Sélectionnez les éléments correspondants dans les deux listes
    src = [rel_src_v[i] for i in indices]
    tgt = [rel_tgt_v[i] for i in indices]

    
    
    with torch.no_grad():
        
        for i,path_txt in enumerate(tgt):
            if i % 1000 == 0:
                print(f"{i} sur {len(tgt)}")
            path = [START_TOKEN] + path_txt.split() + [END_TOKEN]
            int_list = [rel_to_int[src[i]], rel_to_int[SEP_TOKEN]]
            path_sum = 0
            path_tok = 0
            
            for j in range(len(path)-1):
                model.eval()
                int_list.append(rel_to_int[path[j]])
                int_vector = torch.tensor(int_list).unsqueeze(0).to(device)
 
                predictions = model(int_vector)
                
                prob = F.softmax(predictions, dim=1)[0][rel_to_int[path[j+1]]].item()
                
                path_sum += log(prob)
                path_tok += 1
                total_tokens += 1
                sum_prob += log(prob)
                int_list = int_list[1:]

            if i % 100 == 0:
                print(f"Path {i} -- Sum: {path_sum} -- Tokens: {path_tok} -- Perp: {exp(-path_sum / path_tok)}")
    return exp(-sum_prob / total_tokens), total_tokens








def train(model, epochs, dataloader, criterion, optimizer, calculate_perplexity):
    best_perplexity = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    somme = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        k = 0
        for batch in dataloader:
            input_seq = torch.tensor([item[:-1] for item in batch]).to(device)
            target_seq = torch.tensor([item[-1] for item in batch]).to(device)

            if k % 100 == 0:
                print(f"Epoch {epoch}\t\tBatch {k} sur {len(dataloader)}")
            k += 1
            
            model.zero_grad()
            
            logits = model(input_seq)
            loss = criterion(logits, target_seq)

            loss.backward()
            optimizer.step()
            
            running_loss += loss.detach().cpu().numpy()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} - loss: {epoch_loss:.3f}")
        print()
        # Phase d'évaluation
        model.eval()
        val_loss = 0.0
        perplexity, _ = calculate_perplexity(model)
        somme += perplexity
        print(f"Perplexity: {perplexity}")
        # Sauvegarde du modèle si la perplexité est la meilleure
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'transformer/code/best_model.pth')
            print(f"Model saved with perplexity: {perplexity}")
        print()
    model.load_state_dict(torch.load('transformer/code/best_model.pth'))
    print(f"Average perplexity : {somme/epochs}")
    print(f'Best perplexity: {best_perplexity}')
    return model, best_perplexity





if os.path.exists(chemin_t + name_Ngram):
    with open(chemin_t + name_Ngram, 'rb') as f:
        model = pickle.load(f)
else:
    model = NGramTextGen(
        vocab_size=rel_vocab_size,
        embed_dim=embed_dim,
        context_size=n_gram-1,
        hidden_dim=hidden_dim
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #model.load_state_dict(torch.load('transformer/code/best_model_5-12-avec-10-l4.pth'))
    model, bp = train(model, epochs, dataloader, criterion, optimizer, calculate_perplexity)

    open(chemin_t + name_Ngram, 'wb').write(pickle.dumps(model))