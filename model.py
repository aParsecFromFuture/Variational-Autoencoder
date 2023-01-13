import torch
from torch import nn
import config as cfg


class Encoder(nn.Module):
    def __init__(self, inp_dim, hid_dim, z_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(inp_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, z_dim * 2))

    def forward(self, x):
        out = self.model(x)
        mu, log_var = torch.split(out, (out.shape[1] // 2), 1)
        sigma = torch.exp(log_var).pow(0.5)
        eps = torch.randn_like(mu)
        z = mu + sigma * eps
        return z, mu, log_var


class Decoder(nn.Module):
    def __init__(self, z_dim, hid_dim, out_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
            nn.Unflatten(1, (cfg.num_channels, cfg.img_dim, cfg.img_dim)),
            nn.Sigmoid())

    def forward(self, x):
        return self.model(x)


class VAE(nn.Module):
    def __init__(self, inp_dim, hid_dim, z_dim):
        super().__init__()
        self.encoder = Encoder(inp_dim, hid_dim, z_dim)
        self.decoder = Decoder(z_dim, hid_dim, inp_dim)

    def forward(self, x):
        z, mu, log_var = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var

    def generate(self, n):
        self.eval()
        with torch.no_grad():
            mu = torch.zeros(n, cfg.z_dim)
            sigma = torch.ones(n, cfg.z_dim)
            eps = torch.randn(n, cfg.z_dim)
            z = mu + sigma * eps
            return self.decoder(z.to(cfg.device))

