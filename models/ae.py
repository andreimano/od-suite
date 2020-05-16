import torch
import pandas as pd
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

# from od-suite.models.base import VAE
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.functional as F
from typing import Tuple
from tqdm import tqdm


class AnomalyAE(nn.Module):
    """
    """

    def __init__(
        self,
        threshold: float = None,
        score: str = "mse",
        encoder: torch.nn.Sequential = None,
        decoder: torch.nn.Sequential = None,
        bottleneck: int = 5,
        input_dim: int = 768,
        hidden_dim: int = 64,
        no_of_layers: int = 2,
        tensorboard: bool = True,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.score = score

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
                    f"Params: bottleneck:{bottleneck}, input dim:"
                    f"{input_dim}, hidden dim: {hidden_dim}, hidden size: "
                    f"layers: {no_of_layers}"
                )
            )

            encoder_modules = [
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.LeakyReLU(),
            ]
            decoder_modules = [
                nn.Linear(int(bottleneck), int(hidden_dim)),
                nn.LeakyReLU(),
            ]

            for _ in range(no_of_layers - 1):
                encoder_modules.append(nn.Linear(hidden_dim, hidden_dim))
                encoder_modules.append(nn.LeakyReLU())
                decoder_modules.append(nn.Linear(hidden_dim, hidden_dim))
                decoder_modules.append(nn.LeakyReLU())
            encoder_modules.append(nn.Linear(hidden_dim, bottleneck))
            decoder_modules.append(nn.Linear(hidden_dim, input_dim))

            self.encoder = nn.Sequential(*encoder_modules)
            self.decoder = nn.Sequential(*decoder_modules)

        self.device = torch.device("cuda")
        logger.info(f"Device: {self.device}")
        logger.debug(f"Encoder: {self.encoder}")
        logger.debug(f"Decoder: {self.decoder}")

    def calculate_threshold(self):
        pass

    def encode(self, x: torch.tensor) -> torch.tensor:
        encoded = self.encoder(x)
        return x

    def decode(self, z: torch.tensor) -> torch.tensor:
        decoded = self.decoder(z)
        return decoded

    def forward(self, x: torch.tensor) -> torch.tensor:
        x = x.to(self.device)
        z = self.encode(x)
        y = self.decode(z)
        return y

    def loss(self, input, output):
        if self.score == "mse":
            return F.mse_loss(input, output)

    def fit(
        self,
        optim: torch.optim.Adam,
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
                reconstruction = self.forward(batch)
                loss = self.loss(batch, reconstruction)
                loss.backward()
                optim.step()
                loss_value = loss.detach().cpu().item()
                pbar_train.set_description_str(
                    f"(TRAIN) Epoch: {e} Loss: {loss_value:.2f}"
                )
            if self.tensorboard == True:
                self.writer.add_scalar(f"Loss/train", loss_value, e)

            if val_loader is not None:
                self.eval()
                pbar_val = tqdm(val_loader)
                with torch.no_grad():
                    for idx, batch in enumerate(pbar_val):
                        batch = batch.to(self.device)
                        reconstruction, μ, log_σ = self.forward(batch)
                        loss = self.elbo(batch, reconstruction, μ, log_σ, β=1)
                        loss_value = loss.cpu().item()
                        pbar_val.set_description_str(
                            f"(VAL)Epoch: {e} Loss: {loss_value:.2f}"
                        )
                    if self.tensorboard == True:
                        self.writer.add_scalar(f"Loss/valid", loss_value, idx)
