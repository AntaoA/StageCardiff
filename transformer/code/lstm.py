import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import os
import pickle
from module_transformer import TextDataset, TextGen
from lstm_param import chemin_t, device, chemin_data_train, hidden_dim
from lstm_param import SEQUENCE_LENGTH, BATCH_SIZE, epochs, learning_rate, embed_dim, num_layers
from transformer_validation import calculate_perplexity
import numpy as np
import copy

from module_LSTM import LSTMTextGen

name_lstm = "lstm.pickle"

if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_train + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    print("eor: missing data")

rel_vocab_size = len(rel_vocab)

if os.path.exists(chemin_data_train + 'list_path_10.pickle'):
    with open(chemin_data_train + 'list_path_10.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)
else:
    print("Error: missing data")


dataset = TextDataset(samples, rel_to_int)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)

def train(model, epochs, dataloader, criterion, optimizer, calculate_perplexity):
    best_perplexity = np.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0
    somme = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        k = 0
        for input_seq, target_seq, padding_mask in dataloader:
            if k % 1000 == 0:
                print(f"Epoch {epoch}\t\tBatch {k} sur {len(dataloader)}")
            k += 1
            input_seq, target_seq, padding_mask = input_seq.to(device), target_seq.to(device), padding_mask.to(device)

            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, rel_vocab_size)

            active_loss = padding_mask.view(-1) == 1
            active_logits = outputs.view(-1, rel_vocab_size)[active_loss]
            active_labels = target_seq.view(-1)[active_loss]

            loss = criterion(active_logits, active_labels)

            optimizer.zero_grad()
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
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, 'transformer/code/best_model_lstm.pth')
            print(f"Model saved with perplexity: {perplexity}")
        print()
    model.load_state_dict(torch.load('transformer/code/best_model_lstm.pth'))
    print(f"Average perplexity : {somme/epochs}")
    print(f'Best perplexity: {best_perplexity} - epoch {best_epoch}')
    return model, best_perplexity



if os.path.exists(chemin_t + name_lstm):
    model = pickle.load(open(chemin_t + name_lstm, 'rb'))
else:
    model = LSTMTextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers, 
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

    # model.load_state_dict(torch.load('transformer/code/best_model_5-12-avec-10-l4.pth'))
    model, bp = train(model, epochs, dataloader, criterion, optimizer, calculate_perplexity) 

    open(chemin_t + name_lstm, 'wb').write(pickle.dumps(model)) 



 