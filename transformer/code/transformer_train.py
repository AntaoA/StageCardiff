import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import pickle
from module_transformer import TextDataset, TextGen
from transformer_param import chemin_t, device, chemin_data_train
from transformer_param import SEQUENCE_LENGTH, BATCH_SIZE, epochs, learning_rate, embed_dim, num_layers, num_heads


if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_train + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    print("Error: missing data")

rel_vocab_size = len(rel_vocab)

if os.path.exists(chemin_data_train + 'list_path.pickle'):
    with open(chemin_data_train + 'list_path.pickle', 'rb') as f:
        samples, rel_src, rel_tgt = pickle.load(f)
else:
    print("Error: missing data")


dataset = TextDataset(samples, rel_to_int)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)

# Training
def train(model, epochs, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        k = 0
        for input_seq, target_seq, padding_mask in dataloader:
            if k % 100 == 0:
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

if os.path.exists(chemin_t + 'transformer.pickle'):
    with open(chemin_t + 'transformer.pickle', 'rb') as f:
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

    train(model, epochs, dataloader, criterion, optimizer) 
    open(chemin_t + 'transformer.pickle', 'wb').write(pickle.dumps(model))