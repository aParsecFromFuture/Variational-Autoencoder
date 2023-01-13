from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from trainer import Trainer
import config as cfg

dataset = datasets.MNIST(root='.',
                         train=True,
                         download=True,
                         transform=transforms.ToTensor())

dataloader = DataLoader(dataset,
                        batch_size=cfg.batch_size,
                        shuffle=True)

vae_trainer = Trainer(dataloader, debug=True, save_examples=True)
vae_trainer.run()
