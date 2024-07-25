import torch

chemin_t = "transformer/code/"
SEQUENCE_LENGTH = 8
BATCH_SIZE = 56

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 20
learning_rate = 0.00075
embed_dim=200
num_layers=4
dropout=0.3
hidden_dim = 200

chemin_data = "grail-master/data/fb237_v4/"
chemin_data_train = "grail-master/data/fb237_v4/train/"
chemin_data_validation = "grail-master/data/fb237_v4/validation/"
chemin_data_test = "grail-master/data/fb237_v4_ind/test/"

name_lstm = "lstm.pickle"

chemin_t_data = "transformer/data/"
nb_paths_per_triplet = 10

input = False

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
SEP_TOKEN = '<SEP>'
