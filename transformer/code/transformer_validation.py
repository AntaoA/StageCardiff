from transformer_param import chemin_data_validation, chemin_data_train, chemin_t, device, BATCH_SIZE, SEQUENCE_LENGTH
import os
import pickle
import torch
from torch.nn import functional as F
from module_transformer import TextDataset
from torch.utils.data import DataLoader
from math import log

if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_validation + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    print("Erreur: fichier index.pickle non trouvé")

def tokenize(sentence):
    return [rel_to_int[rel] for rel in sentence.split() if rel in rel_to_int]


if os.path.exists(chemin_t + 'transformer.pickle'):
    with open(chemin_t + 'transformer.pickle', 'rb') as f:
        model = pickle.load(f)
else:
    print("Erreur: fichier transformer.pickle non trouvé")


if os.path.exists(chemin_data_validation + 'list_path.pickle'):
    with open(chemin_data_validation + 'list_path.pickle', 'rb') as f:
        samples = pickle.load(f)
else:
    print("Erreur: fichier list_path.pickle non trouvé")


dataset = TextDataset(samples, rel_to_int)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)

def calculate_perplexity(model, dataset):
    model.eval()
    model.to(device)
    
    total_log_likelihood = 0.0
    total_tokens = 0

    with torch.no_grad():
        
        for input_seq, target_seq, padding_mask in dataloader:
            input_seq, target_seq, padding_mask = input_seq.to(device), target_seq.to(device), padding_mask.to(device)
            
            tokenized_inputs = tokenize(input_seq)

            
            outputs = model(input_seq)
            
            for i in range(BATCH_SIZE):
                for j in range(SEQUENCE_LENGTH):
                    tgt_token = target_seq[i][j]
                    prob = outputs[i][j][tgt_token]
                    total_tokens += 1
                    total_log_likelihood += log(prob)
    
    return total_log_likelihood, total_tokens

log_likelihood, tokens = calculate_perplexity(model, dataset)
print(f"Perplexity: {-log_likelihood} -- Tokens: {tokens}")

