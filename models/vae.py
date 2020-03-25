import torch
import pandas as pd
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

# from od-suite.models.base import VAE
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple
from tqdm import tqdm


class AnomalyVAE(nn.Module):
    """
    """

    def __init__(
        self,
        threshold: float = None,
        score: str = "mse",
        samples: int = 10,
        encoder: torch.nn.Sequential = None,
        decoder: torch.nn.Sequential = None,
        beta: float = 1.0,
        latent_dim: int = 2,
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
        else:
            logger.info(f"Using default model.")
            logger.info(
                (
                    f"Params: beta:{beta}, latent dim:{latent_dim}, input dim:"
                    f"{input_dim}, hidden dim: {hidden_dim}, hidden size: "
                    f"layers: {no_of_layers}"
                )
            )

            encoder_modules = [
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.LeakyReLU(),
            ]
            decoder_modules = [
                nn.Linear(int(latent_dim), int(hidden_dim)),
                nn.LeakyReLU(),
            ]

            for _ in range(no_of_layers - 1):
                encoder_modules.append(nn.Linear(hidden_dim, hidden_dim))
                encoder_modules.append(nn.LeakyReLU())
                decoder_modules.append(nn.Linear(hidden_dim, hidden_dim))
                decoder_modules.append(nn.LeakyReLU())
            encoder_modules.append(nn.Linear(hidden_dim, latent_dim))
            decoder_modules.append(nn.Linear(hidden_dim, input_dim))

            self.encoder = nn.Sequential(*encoder_modules)
            self.decoder = nn.Sequential(*decoder_modules)

        self.mu = nn.Linear(self.encoder[-1].out_features, latent_dim)  # type: ignore
        self.log_sigma = nn.Linear(self.encoder[-1].out_features, latent_dim)  # type: ignore

        self.device = torch.device("cuda")
        logger.info(f"Device: {self.device}")
        logger.debug(f"Encoder: {self.encoder}")
        logger.debug(f"Decoder: {self.decoder}")
        logger.debug(f"Mu: {self.mu}")
        logger.debug(f"Log_sigma: {self.log_sigma}")

    def elbo(
        self,
        input: torch.tensor,
        target: torch.tensor,
        mu: torch.tensor,
        log_var: torch.tensor,
        beta: float,
    ) -> torch.tensor:
        CE = F.binary_cross_entropy(input, target, reduction="sum")  # type: ignore
        KL = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return CE + beta * KL

    def reparametrize(self, mu: torch.tensor, log_sigma: torch.tensor) -> torch.tensor:
        sigma = torch.exp(0.5 * log_sigma)
        epsilon = torch.rand_like(sigma)
        return epsilon.mul(sigma).add_(mu)  # type: ignore

    def calculate_threshold(self):
        pass

    def predict_outliers(
        self,
        X: torch.tensor,
        contamination: float,
        reconstruction_metric: torch.nn.Module = torch.nn.MSELoss(),
    ) -> torch.tensor:
        # make n_samples copies of X
        # do inference on the n_samples
        # calculate the reconstruction score flat_score on them
        # instance score is mse, feature score is mse reduction=none
        reconstruction, mu, log_sigma = self.forward(X)
        reconstruction_score = reconstruction_metric(X, reconstruction)
        pass

    def encode(
        self, x: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        encoded = self.encoder(x)
        mu, log_sigma = self.mu(encoded), self.log_sigma(encoded)
        z = self.reparametrize(mu, log_sigma)
        # logger.debug(f"encoded.shape: {encoded.shape}")
        # logger.debug(f"mu.shape: {mu.shape}")
        # logger.debug(f"log_sigma.shape: {log_sigma.shape}")
        # logger.debug(f"z.shape: {z.shape}")
        return z, mu, log_sigma

    def decode(self, z: torch.tensor) -> torch.tensor:
        x = self.decoder(z)
        x = F.sigmoid(x)  # type: ignore
        return x

    def forward(self, input: torch.tensor) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:  # type: ignore
        input = input.to(self.device)
        z, mu, log_sigma = self.encode(input)
        x = F.sigmoid(self.decode(z))  # type: ignore
        return x, mu, log_sigma

    def fit(
        self,
        optim: torch.optim.Adam,  # type: ignore
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        epochs: int = 5,
    ) -> None:
        for e in range(epochs):
            self.train()
            pbar_train = tqdm(train_loader)
            for idx, batch in enumerate(pbar_train):
                optim.zero_grad()
                batch = batch.to(self.device)
                reconstruction, mu, log_sigma = self.forward(batch)
                loss = self.elbo(reconstruction, batch, mu, log_sigma, beta=1)
                loss.backward()
                optim.step()
                loss_value = loss.detach().cpu().item()
                pbar_train.set_description_str(f"(TRAIN)Epoch: {e} Loss: {loss_value:.2f}")
            if self.tensorboard == True:
                self.writer.add_scalar(f"Loss/train", loss_value, e)

            if val_loader is not None:
                self.eval()
                pbar_val = tqdm(val_loader)
                with torch.no_grad():
                    for idx, batch in enumerate(pbar_val):
                        batch = batch.to(self.device)
                        reconstruction, mu, log_sigma = self.forward(batch)
                        loss = self.elbo(batch, reconstruction, mu, log_sigma, beta=1)
                        loss_value = loss.cpu().item()
                        pbar_val.set_description_str(
                            f"(VAL)Epoch: {e} Loss: {loss_value:.2f}"
                        )
                    if self.tensorboard == True:
                        self.writer.add_scalar(f"Loss/valid", loss_value, idx)
