#%%
import sys

sys.path.append("/root/od-suite/")
from models import AnomalyAE
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda")
model = AnomalyAE().cuda()
print("Hello, world")
