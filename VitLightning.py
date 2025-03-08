import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import wandb
from lightning_fabric.utilities import measure_flops
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import medmnist
from medmnist import INFO
from models.MedViT import MedViT_small  # or MedViT_base, MedViT_large as needed

class LitMedViT(pl.LightningModule):
    """ MedViT architecture without classification head so we can do SupConLoss """
    def __init__(self, 
                 n_classes: int, 
                 lr: float, 
                 pretrained_path: str) -> None:
        super().__init__()
        self.save_hyperparameters()  # logs hyperparameters to logger
        self.lr = lr
        self.n_classes = n_classes
        self.model = MedViT_small(num_classes=n_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        # I swap classifier head; forward returns backbone embeddings.
        self.model.proj_head = nn.Identity()
        # Sup-con loss for label informed contrastive loss 
        self.metric_loss_func = losses.SupConLoss(temperature=0.07)
        
    def forward(self, x):
        # I use L2 normalized embeddings as suggested in SupCon paper
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        inputs, targets = batch
        embeddings = self.forward(inputs)
        loss = self.metric_loss_func(embeddings, targets.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        inputs, targets = batch
        embeddings = self.forward(inputs)
        return {"embeddings": embeddings.cpu(), 
                "targets": targets.cpu()}
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.lr, 
                              momentum=0.9)
        return optimizer
