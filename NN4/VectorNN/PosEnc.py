import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import os
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.onnx

# Positional encoding is modeled after the functions of sin and cos of (k/(n^((2i)/d))). k represents the position of the word in the sentence, n is set to 10,000 (I don't know
# what n is), d is the dimensionality of the model (which the paper uses 512), and i represents an index which refers to the individual dimensions within the embedding. 