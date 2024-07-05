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


if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_train + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, _, _, _ = pickle.load(f)

rel_vocab_size = len(rel_vocab)

if os.path.exists(chemin_data_train + 'list_path.pickle'):
    with open(chemin_data_train + 'list_path.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)

if os.path.exists(chemin_data_validation + 'index.pickle'):
    with open(chemin_data_validation + 'index.pickle', 'rb') as f:
        int_to_rel_v, rel_to_int_v, rel_vocab_v, _, _, _ = pickle.load(f)

if os.path.exists(chemin_data_validation + 'list_path.pickle'):
    with open(chemin_data_validation + 'list_path.pickle', 'rb') as f:
        samples_v, rel_src_v, rel_tgt_v = pickle.load(f)

#same data as in transformer_train.py
new_samples = []

for sample in samples:
    src, tgt = sample[2:], sample[0]
    if len(src) < SEQUENCE_LENGTH-2:
        src = src + [PAD_TOKEN] * (SEQUENCE_LENGTH - len(src) - 2)
    new_samples.append(src + [SEP_TOKEN, tgt])

new_samples_v = []
for sample in samples_v:
    src, tgt = sample[2:], sample[0]
    if len(src) < SEQUENCE_LENGTH-2:
        src = src + [PAD_TOKEN] * (SEQUENCE_LENGTH - len(src) - 2)
    new_samples_v.append(src + [SEP_TOKEN, tgt])

dataset = TextDataset(new_samples, rel_to_int)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)



def calculate_perplexity(model):
    model.eval()
    model.to(device)

    sum_prob = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, sample in enumerate(new_samples_v):
            if i % 1000 == 0:
                print(f"{i} sur {len(new_samples_v)}")
            tgt = rel_src_v[0]
            int_list = [rel_to_int[rel] for rel in sample[:-1]]
            int_list.append(rel_to_int[SEP_TOKEN])
        
            model.eval()
            int_vector = torch.tensor(int_list).unsqueeze(0).to(device)

            predictions = model(int_vector)
                    
            prob = F.softmax(predictions[:, -1, :], dim=-1)[0][rel_to_int[tgt]].item()
                

            total_tokens += 1
            sum_prob += log(prob)

    return exp(-sum_prob / total_tokens), total_tokens




if os.path.exists(chemin_t + 'classifier.pickle'):
    with open(chemin_t + 'classifier.pickle', 'rb') as f:
        model = pickle.load(f)
else:
    model = TextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=embed_dim,
        num_layers=num_layers, 
        num_heads=num_heads,
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    train(model, epochs, dataloader, criterion, optimizer, calculate_perplexity) 
    open(chemin_t + 'classifier.pickle', 'wb').write(pickle.dumps(model))

def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([rel_to_int[word] for word in words[-SEQUENCE_LENGTH:]]).unsqueeze(0)
    return input_seq

def sample_next(predictions, k):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    topk_probs, topk_indices = torch.topk(probabilities, k)
    return topk_probs[0], topk_indices[0]

def text_generator(sentence, generate_length):
    model.eval()
    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample)
        if len(int_vector) >= SEQUENCE_LENGTH - 1:
            break
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions, 1)
        sample += ' ' + int_to_rel[next_token]
        if next_token == rel_to_int[END_TOKEN]:
            break
    print(sample)
    print('\n')


def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([rel_to_int[word] for word in words]).unsqueeze(0)
    return input_seq

def sample_next(predictions, k):
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    topk_probs, topk_indices = torch.topk(probabilities, k)
    return topk_probs[0], topk_indices[0]

def text_generator(sentence):
    model.eval()
    sample = sentence
    int_vector = return_int_vector(sample)
    input_tensor = int_vector.to(device)
    with torch.no_grad():
        predictions = model(input_tensor)
    next_token = sample_next(predictions, 1)
    print(int_to_rel[next_token])
    print('\n')

    
def text_generator_with_confidence(sentence, k):
    model.eval()
    int_vector = return_int_vector(sentence).to(device)
    candidates = []
    with torch.no_grad():
        predictions = model(int_vector)
        
    topk_probs, topk_indices = sample_next(predictions, k)
        
    for i in range(k):
        next_token_index = topk_indices[i].item()
        next_token_prob = topk_probs[i].item()
        next_token = int_to_rel[next_token_index]
        candidates.append((next_token, next_token_prob))
    
    return candidates





lines = []

if os.path.exists(chemin_t_data + 'nouvelles_relations.txt'):
    with open(chemin_t_data + 'nouvelles_relations.txt', 'rb') as f:
        lines += f.readlines()
else:
    print("Rien à classifier\n")

#data to classify
rel_src = []
rel_tgt = []

for line in lines:
    pred, prob, prob_l = line.strip().decode('utf-8').split(":")
    if float(prob) > 0.05:
        prediction = pred.split()
        rel_src.append(prediction[2:])
        rel_tgt.append(prediction[0])


with open(chemin_t_data + "classification_path.txt", "w") as f:
    for i, path in enumerate(rel_src):
            sentence = " ".join(path) + " <SEP>"
            out = text_generator_with_confidence(sentence, 5)
            for r, p in out:
                f.write(f"{path} : {rel_tgt[i]} : {r} : {p} \n")