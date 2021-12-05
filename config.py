# These libraries help to interact with the operating system and the runtime environment respectively
import os
import sys

# Model/Training related libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# Dataloader libraries
from torch.utils.data import DataLoader, Dataset

# Transforms and datasets
import torchvision.transforms as transforms
import torchvision.datasets as dset

import time
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Allosaurus
from allosaurus.audio import read_audio
from allosaurus.app import read_recognizer
from allosaurus.am.utils import *


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device