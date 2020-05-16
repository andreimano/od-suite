import sys

sys.path.append("/root/od-suite/")
from models import AnomalyVAE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda")


class KDDData(Dataset):
    def __init__(self, data):
        self.data = data.to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


model = AnomalyVAE().cuda()

x = torch.randn(768).cuda()
x = x.reshape(1, 768)

logger.info("(test) x.shape:", x.shape)

with torch.no_grad():
    y, _, _ = model.forward(x)

    mse = torch.nn.MSELoss().cuda()

    logger.info(f"x.shape:{x.shape}")
    logger.info(f"y.shape:{y.shape}")
    loss = mse(x, y)
    logger.info(f"mse:{loss}")


data_rand = torch.randn(20, 768)

dataset = KDDData(data_rand)

print("noseg3")

dataloader = DataLoader(dataset, batch_size=4)

print("noseg4")

optim = torch.optim.Adam(params=model.parameters())

model.fit(optim, dataloader, epochs=20)

with torch.no_grad():
    y, _, _ = model.forward(x)

    mse = torch.nn.MSELoss().cuda()

    logger.info(f"x.shape:{x.shape}")
    logger.info(f"y.shape:{y.shape}")
    loss = mse(x, y)
    logger.info(f"mse:{loss}")
