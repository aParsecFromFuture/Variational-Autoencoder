import torch
from torchvision.utils import make_grid
from model import VAE
from utils import save
import config as cfg


class Trainer:
    def __init__(self, dataloader, debug=True, save_path=None, save_examples=True):
        super().__init__()

        self.debug = debug
        self.save_path = save_path
        self.save_examples = save_examples
        self.model = VAE(cfg.inp_dim, cfg.hid_dim, cfg.z_dim).to(cfg.device)
        self.dataloader = dataloader
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def run(self):
        self.model.train()
        for epoch in range(cfg.num_epochs):
            for idx, (x, _) in enumerate(self.dataloader):
                x = x.to(cfg.device)
                x_hat, mu, log_var = self.model(x)

                kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / cfg.batch_size
                recon_loss = torch.sum((x - x_hat).pow(2)) / cfg.batch_size
                loss = kl_div + recon_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if self.debug:
                print(f'Epoch: {epoch}, RLoss: {recon_loss.item():.3f}, KLLoss: {kl_div.item():.3f}')

        if self.save_path is not None:
            torch.save(self.model.state_dict(), self.save_path)
            print(f'Model saved to {self.save_path}!')

        if self.save_examples:
            imgs_recon = self.model.generate(64)
            save(make_grid(imgs_recon.cpu()))
            print(f'Images saved!')

