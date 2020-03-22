import torch
import pandas as pd
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

# from od-suite.models.base import VAE
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as F
from typing import Tuple


class AnomalyVAE(nn.Module):
    """
    """
    def __init__(
        self,
        threshold: float = None,
        score: str = "mse",
        samples=10,
        encoder: torch.nn.Sequential = None,
        decoder: torch.nn.Sequential = None,
        beta: float = 1.0,
        latent_dim: int = 1,
        input_dim: int = 768,
        hidden_dim: int = 64,
        no_of_layers: int = 2,
        tensorboard: bool = True,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.score = score
        self.samples = samples

        self.tensorboard = tensorboard
        if tensorboard == True:
            self.writer = SummaryWriter()

        if isinstance(encoder, torch.nn.Sequential) and isinstance(
            decoder, torch.nn.Sequential
        ):
            logger.info(f"Using encoder and decoder.")
            self.encoder = encoder
            self.decoder = decoder
            self.mu = nn.Linear(encoder[-1].out_features, latent_dim) #type: ignore
            self.sigma = nn.Linear(hidden_dim, decoder[0].in_features) #type: ignore

        else:
            logger.info(f"Using default model.")
            logger.info(
                "Params: beta:{beta}, latent dim:{latent_dim}, input dim: {input_dim}, hidden dim: {hidden_dim}, hidden size: {hidden_size}, layers: {no_of_layers}"
            )
            self.mu = nn.Linear(int(hidden_dim), int(latent_dim))
            self.log_sigma = nn.Linear(int(hidden_dim), int(latent_dim))

            encoder_modules = [nn.Linear(int(input_dim), int(hidden_dim)), nn.LeakyReLU()]
            decoder_modules = [nn.Linear(int(latent_dim), int(hidden_dim)), nn.LeakyReLU()]

            for _ in range(no_of_layers - 1):
                encoder_modules.append(nn.Linear(hidden_dim, hidden_dim))
                encoder_modules.append(nn.LeakyReLU())
                decoder_modules.append(nn.Linear(hidden_dim, hidden_dim))
                decoder_modules.append(nn.LeakyReLU())
            encoder_modules.append(nn.Linear(hidden_dim, latent_dim))
            decoder_modules.append(nn.Linear(latent_dim, input_dim))

            self.encoder = nn.Sequential(*encoder_modules)
            self.decoder = nn.Sequential(*decoder_modules)

        self.device = torch.device("cuda")

    def elbo(self, input: torch.Tensor, target: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: int) -> torch.Tensor:
        CE = F.binary_cross_entropy(input, target, reduction='sum') #type: ignore
        KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return CE + beta * KL
    
    def reparametrize(self, mu: torch.Tensor, log_sigma: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(0.5 * log_sigma)
        epsilon = torch.rand_like(sigma)
        return epsilon.mul(sigma).add_(mu) #type: ignore
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(x)
        mu, log_sigma = self.mu(encoded), self.log_sigma(encoded)
        z = self.reparametrize(mu, log_sigma)
        return z, mu, log_sigma
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        x = self.decoder(z)
        x = F.sigmoid(x) #type: ignore
        return x

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: #type: ignore
        z, mu, log_sigma = self.encode(input)
        x = self.decode(z)
        return x, mu, log_sigma

    def fit(
        self,
        optim: torch.nn.optim.Adam, #type: ignore
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 30,
    ):
        for e in range(epochs):
            self.train()
            for idx, batch in enumerate(train_loader):
                optim.zero_grad()
                batch = batch.to(self.device)
                reconstruction, mu, log_sigma = self.forward(batch)
                loss = self.elbo(batch, reconstruction, mu, log_sigma, beta=1)
                loss.backward()
                optim.step()
                if self.tensorboard == True:
                    self.writer.add_scalar(
                        f"Loss/train/e{e}", loss.detach().cpu().item(), idx
                    )

            self.eval()
            with torch.no_grad():
                for idx, batch in enumerate(val_loader):
                    batch = batch.to(self.device)
                    reconstruction, mu, log_sigma = self.forward(batch)
                    loss = self.elbo(batch, reconstruction, mu, log_sigma, beta=1)
                    if self.tensorboard == True:
                        self.writer.add_scalar(
                            f"Loss/valid/e{e}", loss.cpu().item(), idx
                        )

