from torch.utils.data import DataLoader
from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK
from module_transformer import TextDataset, TextGen
from transformer_train import train
import torch.nn as nn
import torch.optim as optim
from transformer_param import SEQUENCE_LENGTH, device, chemin_data_train
from transformer_validation import calculate_perplexity
import os
import pickle

if os.path.exists(chemin_data_train + 'index.pickle'):
    with open(chemin_data_train + 'index.pickle', 'rb') as f:
        int_to_rel, rel_to_int, rel_vocab, vocab_input, rel_to_int_input, int_to_rel_input = pickle.load(f)
else:
    print("Error: missing data")

rel_vocab_size = len(rel_vocab)

if os.path.exists(chemin_data_train + 'list_path.pickle'):
    with open(chemin_data_train + 'list_path.pickle', 'rb') as f:
        samples = pickle.load(f)
else:
    print("Error: missing data")
    

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
    perplexity, n_tokens = calculate_perplexity(model)
    # Minimiser la perte moyenne
    return {'loss': perplexity, 'status': STATUS_OK}

# Définir les essais pour stocker les résultats
trials = Trials()

# Exécuter l'optimisation
best = fmin(fn=objective, space=space, algo=rand.suggest, max_evals=20, trials=trials)

print("Best hyperparameters found:", best)