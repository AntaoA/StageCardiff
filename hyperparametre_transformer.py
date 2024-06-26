


    from hyperopt import hp, fmin, tpe, Trials, space_eval

    
    
    
    
    
EPOCH_v = [10,100]
#BATCH_SIZE_v = [4,1024]
OPTIMIZER_v = ['rmsprop', 'adam']
NUM_LAYERS_v = [1,16]
NUM_HEADS_v = [2,16]
EMBED_DIM_v = [1,512]
LEARNING_RATE_v = [10e-6,10e-1]

def objective(params):
    epochs = params['epochs']
    #batch_size = params['batch_size']
    optimizer = params['optimizer']
    num_layers = params['num_layers']
    num_heads = params['num_heads']
    embed_dim = params['embed_dim']
    learning_rate = params['learning_rate']
    
    # Update the model and training parameters
    model = t.TextGen(
        vocab_size=rel_vocab_size, 
        embed_dim=embed_dim,
        num_layers=num_layers, 
        num_heads=num_heads,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError("Invalid optimizer")
    
    # Train the model
    train(model, epochs, dataloader, criterion, optimizer)
    
    # Calculate the loss on a validation set and return it as the objective value
    # Replace `validation_set` with your actual validation data
    validation_loss = calculate_validation_loss(model, validation_set)
    
    return validation_loss

# Define the search space
space = {
    'epochs': hp.choice('epochs', EPOCH_v),
    #'batch_size': hp.choice('batch_size', BATCH_SIZE_v),
    'optimizer': hp.choice('optimizer', OPTIMIZER_v),
    'num_layers': hp.choice('num_layers', NUM_LAYERS_v),
    'num_heads': hp.choice('num_heads', NUM_HEADS_v),
    'embed_dim': hp.choice('embed_dim', EMBED_DIM_v),
    'learning_rate': hp.loguniform('learning_rate', np.log(LEARNING_RATE_v[0]), np.log(LEARNING_RATE_v[1]))
}

# Run the hyperparameter optimization
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
