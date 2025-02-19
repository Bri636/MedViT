
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from utils import BaseConfig

class KNNCallBackConfig(BaseConfig): 
    k: int = 5
    """ k (int): Number of neighbors for k-NN."""
    log_every_n_steps: int = 5
    """ log_every_n_steps (int): Frequency of evaluation during training."""
    max_train_batches: int = 10
    """ 
    max_train_batches (int): Number of validation batches to use for evaluation so we dont compute for 
    all validation data
    """

class KNN_Evaluation_Callback(Callback):
    def __init__(self, 
                 train_dataloader, 
                 val_dataloader, 
                 k=5, 
                 log_every_n_steps=100, 
                 max_train_batches=10):
        """
        Args:
            train_dataloader: Dataloader for training data.
            val_dataloader: Dataloader for validation data.
            k (int): Number of neighbors for k-NN.
            log_every_n_steps (int): Frequency of evaluation during training.
            max_train_batches (int): Number of training batches to use for evaluation.
        """
        super().__init__()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.k = k
        self.log_every_n_steps = log_every_n_steps
        self.batch_count = 0
        self.max_train_batches = max_train_batches  # Use only this many batches from train set

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        if self.batch_count % self.log_every_n_steps == 0:
            print(f"Running Evaluation at global step {trainer.global_step}...")
            self.run_knn_evaluation(trainer, pl_module)

    def run_knn_evaluation(self, trainer, pl_module):
        pl_module.eval()  # Set model to eval mode
        device = pl_module.device

        # Use only a subset of the training dataloader
        train_embeddings = []
        train_labels = []
        for i, t_batch in enumerate(self.train_dataloader):
            if i >= self.max_train_batches:
                break
            inputs, targets = t_batch
            inputs = inputs.to(device)
            with torch.no_grad():
                emb = pl_module(inputs)
            train_embeddings.append(emb.cpu())
            train_labels.append(targets.cpu())
        X_train = torch.cat(train_embeddings, dim=0).numpy()
        y_train = torch.cat(train_labels, dim=0).squeeze().numpy()

        # Compute embeddings on the full validation set
        val_embeddings = []
        val_labels = []
        for v_batch in self.val_dataloader:
            inputs, targets = v_batch
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
            if len(np.unique(y_val)) == 2:
                auc = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
        except Exception:
            auc = float('nan')

        # Generate a confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 6))
        cm = confusion_matrix(y_val, y_pred)
        cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        fig.colorbar(cax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=12)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')

        # Generate t-SNE visualization of validation embeddings
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42)
        X_val_2d = tsne.fit_transform(X_val)
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        scatter = ax2.scatter(X_val_2d[:, 0], X_val_2d[:, 1], c=y_val, cmap='viridis', alpha=0.7)
        ax2.set_title("t-SNE Visualization of Validation Embeddings")
        ax2.set_xlabel("t-SNE component 1")
        ax2.set_ylabel("t-SNE component 2")
        # Set discrete ticks based on the unique labels in y_val
        classes = np.unique(y_val)
        cbar = fig2.colorbar(scatter, ax=ax2, ticks=classes)
        cbar.ax.set_yticklabels(classes)

        # Log metrics and the confusion matrix & embeddings visualization images to WandB.
        log_dict = {
            "KNN Accuracy": acc,
            "KNN F1 Score": f1,
            "KNN AUC": auc,
            "Confusion Matrix": wandb.Image(fig),
            "Embeddings Visualization": wandb.Image(fig2)
        }
        trainer.logger.experiment.log(log_dict, step=trainer.global_step)
        plt.close(fig)
        plt.close(fig2)
        pl_module.train()  # Switch back to training mode
        
