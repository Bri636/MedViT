import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.utilities import measure_flops

from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import medmnist
from medmnist import INFO
from models.MedViT import MedViT_small  # or MedViT_base, MedViT_large as needed
from utils import BaseConfig

# ------------------------------
# Custom Callbacks for k-NN Evaluation & Logging to WandB
# ------------------------------

class KNN_Evaluation_Callback(Callback):
    def __init__(self, train_dataloader, val_dataloader, k=5):
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.k = k
    
    def on_validation_epoch_end(self, trainer, pl_module):
        pl_module.eval()
        device = pl_module.device
        
        # Compute embeddings on the (subset of) training set
        train_embeddings = []
        train_labels = []
        for batch in self.train_dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            with torch.no_grad():
                emb = pl_module(inputs)
            train_embeddings.append(emb.cpu())
            train_labels.append(targets.cpu())
        X_train = torch.cat(train_embeddings, dim=0).numpy()
        y_train = torch.cat(train_labels, dim=0).squeeze().numpy()
        
        # Compute embeddings on the validation set
        val_embeddings = []
        val_labels = []
        for batch in self.val_dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            with torch.no_grad():
                emb = pl_module(inputs)
            val_embeddings.append(emb.cpu())
            val_labels.append(targets.cpu())
        X_val = torch.cat(val_embeddings, dim=0).numpy()
        y_val = torch.cat(val_labels, dim=0).squeeze().numpy()
        
        # Fit a k-NN classifier on the training embeddings
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        
        # Compute metrics
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        try:
            y_proba = knn.predict_proba(X_val)
            # For binary classification use the probability for the positive class;
            # otherwise, use one-vs-rest
            if len(np.unique(y_val)) == 2:
                auc = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
        except Exception as e:
            auc = float('nan')
        
        # Generate a confusion matrix plot
        cm = confusion_matrix(y_val, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', fontsize=12)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Log metrics and the confusion matrix image to WandB.
        log_dict = {
            "KNN Accuracy": acc,
            "KNN F1 Score": f1,
            "KNN AUC": auc,
            "Confusion Matrix": wandb.Image(fig)
        }
        # Log using the experiment associated with the WandbLogger.
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        plt.close(fig)
        pl_module.train()
         
class FlopsLoggerCallback(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        # Ensure the logger is a WandbLogger
        if not isinstance(trainer.logger, WandbLogger):
            raise TypeError("Trainer logger must be a WandbLogger instance.")
        # Create a meta-device model for FLOPs calculation
        with torch.device('meta'):
            meta_model = pl_module.__class__(*pl_module.args, **pl_module.kwargs)
        # Generate a sample input tensor with the same shape as your data
        input_sample = torch.randn(1, 3, 224, 224, device='meta')  # Adjust dimensions as needed
        # Calculate FLOPs using the measure_flops utility
        flops = measure_flops(meta_model, input_sample)
        # Log FLOPs to Weights & Biases
        trainer.logger.experiment.config.update({"FLOPs": flops})