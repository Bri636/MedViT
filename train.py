""" My training implementation for NTXent-like loss with KMeans Clustering """

from __future__ import annotations

import os
import torch
from torch import nn, optim
from tqdm import tqdm
from pytorch_metric_learning import losses  # using SupConLoss for contrastive training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from argparse import ArgumentParser

# packages
import medmnist
from medmnist import INFO, Evaluator
from MedViT import MedViT_small

