import torch

chemin_t = "transformer/code/"
SEQUENCE_LENGTH = 8
BATCH_SIZE = 40

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 100
learning_rate = 0.001 
embed_dim=2400
num_layers=6
num_heads=4


chemin_data = "grail-master/data/fb237_v4/"
chemin_t_data = "transformer/data/"
nb_paths = 5000


START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
SEP_TOKEN = '<SEP>'
