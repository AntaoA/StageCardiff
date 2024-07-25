import torch

chemin_t = "transformer/code/"
SEQUENCE_LENGTH = 8
BATCH_SIZE = 300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
learning_rate = 0.0008
embed_dim=256
num_layers=4
num_heads=4
dropout=0.3
n_gram = 4
hidden_dim = 100

chemin_data = "grail-master/data/fb237_v4/"
chemin_data_train = "grail-master/data/fb237_v4/train/"
chemin_data_validation = "grail-master/data/fb237_v4/validation/"
chemin_data_test = "grail-master/data/fb237_v4_ind/test/"

name_transformer = "transformer_5-29.pickle"

chemin_t_data = "transformer/data/"
nb_paths_per_triplet = 10
nb_paths_per_triplet_fine_tuning = 100

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
SEP_TOKEN = '<SEP>'
