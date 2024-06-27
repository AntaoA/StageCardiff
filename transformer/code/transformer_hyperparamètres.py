from torch.utils.data import DataLoader
from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK
from module_transformer import TextDataset, TextGen
from transformer_import_data import rel_vocab_size, rel_to_int, samples
from transformer_train import train
import torch
import torch.nn as nn
import torch.optim as optim
from transformer_param import SEQUENCE_LENGTH, device


# Espace de recherche des hyperparamètres
space = {
    'epochs': hp.quniform('epochs', 10, 100, 10),
    'batch_size': hp.quniform('batch_size', 8, 128, 8),
    'num_layers': hp.quniform('num_layers', 1, 8, 1),
    'num_heads': hp.quniform('num_heads', 1, 8, 1),
    'embed_dim': hp.quniform('embed_dim', 10, 100, 10),
    'learning_rate': hp.uniform('learning_rate', 0.0001, 0.01)
}
def objective(params):

    # Afficher les paramètres de l'essai en cours
    print("Testing parameters:", params)
    

    # Charger les données
    dataset = TextDataset(samples, rel_to_int)
    dataloader = DataLoader(dataset, batch_size=int(params['batch_size']), shuffle=True)    
    
    # Définir le modèle
    model = TextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=int(params['embed_dim']*params['num_heads']),
        num_layers=int(params['num_layers']), 
        num_heads=int(params['num_heads']),
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    
    # Définir le critère et l'optimisateur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    # Entraînement
    train(model, int(params['epochs']), dataloader, criterion, optimizer)
    # Calculer la perte (ou une autre métrique) pour évaluer le modèle
    # Utiliser une simple moyenne de la perte d'entraînement pour cet exemple
    running_loss = 0
    model.eval()
    with torch.no_grad():
        for input_seq, target_seq, padding_mask in dataloader:
            input_seq, target_seq, padding_mask = input_seq.to(device), target_seq.to(device), padding_mask.to(device)
            outputs = model(input_seq)
            target_seq = target_seq.contiguous().view(-1)
            outputs = outputs.view(-1, rel_vocab_size)
            active_loss = padding_mask.view(-1) == 1
            active_logits = outputs.view(-1, rel_vocab_size)[active_loss]
            active_labels = target_seq.view(-1)[active_loss]
            loss = criterion(active_logits, active_labels)
            running_loss += loss.item()
    avg_loss = running_loss / len(dataloader)
    
    # Minimiser la perte moyenne
    return {'loss': avg_loss, 'status': STATUS_OK}

# Définir les essais pour stocker les résultats
trials = Trials()

# Exécuter l'optimisation
best = fmin(fn=objective, space=space, algo=rand.suggest, max_evals=20, trials=trials)

print("Best hyperparameters found:", best)