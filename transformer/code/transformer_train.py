import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
import transformer.code.module_transformer as t
from transformer.code.module_transformer import START_TOKEN, END_TOKEN, PAD_TOKEN, SEP_TOKEN, SEQUENCE_LENGTH
from transformer.code.module_transformer import TextDataset, TextGen
from transformer.code.transformer_import_data import rel_vocab, rel_vocab_size, rel_to_int, int_to_rel, samples


chemin_t = "transformer/code/"

BATCH_SIZE = 40

dataset = TextDataset(samples, rel_to_int)
dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Training
def train(model, epochs, dataloader, criterion, optimizer):
    model.train()
    for epoch in range(epochs):
        running_loss = 0
        for input_seq, target_seq, padding_mask in dataloader:
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
        print(f"Epoch {epoch} loss: {epoch_loss:.3f}")



epochs = 100
learning_rate = 0.001 
embed_dim=2400
num_layers=6
num_heads=4



if os.path.exists(chemin_t + 'transformer.pickle'):
    with open(chemin_t + 'transformer.pickle', 'rb') as f:
        model = pickle.load(f)
else:
    model = TextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=100,
        num_layers=2, 
        num_heads=2,
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

def return_int_vector(text):
    words = text.split()
    input_seq = torch.LongTensor([rel_to_int[word] for word in words[-t.SEQUENCE_LENGTH:]]).unsqueeze(0)
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
