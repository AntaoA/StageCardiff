import torch

chemin_t = "transformer/code/"
SEQUENCE_LENGTH = 8
BATCH_SIZE = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
learning_rate = 0.001 
embed_dim=32
num_layers=2
num_heads=2

chemin_data = "grail-master/data/fb237_v4/"
chemin_data_train = "grail-master/data/fb237_v4/train/"
chemin_data_validation = "grail-master/data/fb237_v4/validation/"

chemin_t_data = "transformer/data/"
nb_paths_per_triplet = 1


START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
SEP_TOKEN = '<SEP>'
