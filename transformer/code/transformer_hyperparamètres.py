from torch.utils.data import DataLoader
from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK
from module_transformer import TextDataset, TextGen
from transformer_train import train
import torch.nn as nn
import torch.optim as optim
from transformer_param import SEQUENCE_LENGTH, device, chemin_data_train
from transformer_validation import calculate_perplexity
from transformer_import_data import rel_vocab, rel_to_int
import os
import pickle


rel_vocab_size = len(rel_vocab)

if os.path.exists(chemin_data_train + 'list_path.pickle'):
    with open(chemin_data_train + 'list_path.pickle', 'rb') as f:
        samples, _, _ = pickle.load(f)
else:
    print("Error: missing data")
    

# Espace de recherche des hyperparamètres
space = {
    'epochs': hp.quniform('epochs', 10, 100, 1),
    'batch_size': hp.quniform('batch_size', 16, 128, 1),
    'num_layers': hp.quniform('num_layers', 1, 12, 1),
    'num_heads': hp.quniform('num_heads', 1, 16, 1),
    'embed_dim_multiplier': hp.quniform('embed_dim', 2, 32, 2),
    'learning_rate': hp.uniform('learning_rate', 0.00001, 0.1)
}



def objective(params):

    params['embed_dim'] = params['embed_dim_multiplier'] * params['num_heads']


    # Afficher les paramètres de l'essai en cours
    print("Testing parameters:", params)
    

    # Charger les données
    dataset = TextDataset(samples, rel_to_int)
    dataloader = DataLoader(dataset, batch_size=int(params['batch_size']), shuffle=True)    
    
    # Définir le modèle
    model = TextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=int(params['embed_dim']),
        num_layers=int(params['num_layers']), 
        num_heads=int(params['num_heads']),
        sequence_length=SEQUENCE_LENGTH
    ).to(device)
    
    # Définir le critère et l'optimisateur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    # Entraînement
    m, best_perplexity = train(model, int(params['epochs']), dataloader, criterion, optimizer, calculate_perplexity)
    # Calculer la perte (ou une autre métrique) pour évaluer le modèle
    # Utiliser une simple moyenne de la perte d'entraînement pour cet exemple
    print("Perplexity:", best_perplexity)
    # Minimiser la perte moyenne
    return {'perplexity': best_perplexity, 'status': STATUS_OK}

# Définir les essais pour stocker les résultats
trials = Trials()

# Exécuter l'optimisation
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best hyperparameters found:", best)