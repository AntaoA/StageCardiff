from transformer_param import chemin_data_validation, chemin_data_train, chemin_t, device, BATCH_SIZE, SEQUENCE_LENGTH, PAD_TOKEN, SEP_TOKEN, START_TOKEN, END_TOKEN
import os
import pickle
import torch
from torch.nn import functional as F
from math import log, exp


if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_train + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    print("Erreur: fichier index.pickle non trouvé")

if os.path.exists(chemin_data_validation + 'list_path.pickle'):
    with open(chemin_data_validation + 'list_path.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)
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

    with torch.no_grad():
        
        for i,path_txt in enumerate(rel_tgt):
            if i % 1000 == 0:
                print(f"{i} sur {len(rel_tgt)}")
            path = path_txt.split()
            int_list = [rel_to_int[rel_src[i]], rel_to_int[SEP_TOKEN]]

            for j in range(len(path)-1):
                model.eval()
                int_list.append(rel_to_int[path[j]])
                int_vector = torch.tensor(int_list).unsqueeze(0).to(device)

                predictions = model(int_vector)
                
                prob = F.softmax(predictions[:, -1, :], dim=-1)[0][rel_to_int[path[j+1]]].item()
                
                
                total_tokens += 1
                sum_prob += log(prob)

    return exp(-sum_prob / total_tokens), total_tokens


def import_model():
    if os.path.exists(chemin_t + 'transformer.pickle'):
        with open(chemin_t + 'transformer.pickle', 'rb') as f:
            model = pickle.load(f)
            return model
    else:
        print("Erreur: fichier transformer.pickle non trouvé")







#perplexity, tokens = calculate_perplexity(model)
#print(f"Perplexity: {perplexity} -- Tokens: {tokens}") 