import torch

img_dim = 28
num_channels = 1
inp_dim = 28 * 28 * 1
hid_dim = 512
z_dim = 8

lr = 1e-3
num_epochs = 50
batch_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
