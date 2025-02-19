import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import medmnist
from medmnist import INFO

# me packages 
from train.knn_callback import KNNCallBackConfig, KNN_Evaluation_Callback
from medmnist_datasets.medmnist_dataset import make_datasets
from models.MedViT import MedViT_small  # or MedViT_base, MedViT_large as needed
from utils import BaseConfig

# some defaults yeh
NUM_EPOCHS = 10
LR = 0.005

class TrainConfig(BaseConfig): 
    """ Training config """
    model_checkpoint_path: str 
    num_epochs: int = 10 
    lr: float = 0.005

class LitMedViT(pl.LightningModule):
    """ I rip off the classifier head so we can do SupConLoss """
    def __init__(self, 
                 n_classes: int, 
                 lr: float, 
                 pretrained_path: str) -> None:
        super().__init__()
        self.save_hyperparameters()  # logs hyperparameters to logger
        self.lr = lr
        self.n_classes = n_classes
    
        # TODO: lightning will override from current checkpoint; maybe route based on continue=True or somethin
        self.model = MedViT_small(num_classes=n_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        # I rip off classifier head; forward returns backbone embeddings.
        self.model.proj_head = nn.Identity()
        # I use sup-con loss for label informed contrastive loss 
        self.metric_loss_func = losses.SupConLoss(temperature=0.07)
        
    def forward(self, x):
        # I use L2 normalized embeddings as suggested in original paper
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)
    
    def training_step(self, batch: torch.Tensor, batch_idx: int):
        inputs, targets = batch
        embeddings = self.forward(inputs)
        # Ensure targets are 1D (in case they come in shape [B, 1])
        loss = self.metric_loss_func(embeddings, targets.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        # In this example, validation_step does not compute a loss since k-NN evaluation is done in the callback.
        inputs, targets = batch
        embeddings = self.forward(inputs)
        return {"embeddings": embeddings.cpu(), 
                "targets": targets.cpu()}
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), 
                              lr=self.lr, 
                              momentum=0.9)
        return optimizer

# class KNNCallBackConfig(BaseConfig): 
#     k: int = 5
#     """ k (int): Number of neighbors for k-NN."""
#     log_every_n_steps: int = 5
#     """ log_every_n_steps (int): Frequency of evaluation during training."""
#     max_train_batches: 10
#     """ 
#     max_train_batches (int): Number of validation batches to use for evaluation so we dont compute for 
#     all validation data
#     """

# class KNN_Evaluation_Callback(Callback):
#     def __init__(self, 
#                  train_dataloader, 
#                  val_dataloader, 
#                  k=5, 
#                  log_every_n_steps=100, 
#                  max_train_batches=10):
#         """
#         Args:
#             train_dataloader: Dataloader for training data.
#             val_dataloader: Dataloader for validation data.
#             k (int): Number of neighbors for k-NN.
#             log_every_n_steps (int): Frequency of evaluation during training.
#             max_train_batches (int): Number of training batches to use for evaluation.
#         """
#         super().__init__()
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.k = k
#         self.log_every_n_steps = log_every_n_steps
#         self.batch_count = 0
#         self.max_train_batches = max_train_batches  # Use only this many batches from train set

#     def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
#         self.batch_count += 1
#         if self.batch_count % self.log_every_n_steps == 0:
#             print(f"Running Evaluation at global step {trainer.global_step}...")
#             self.run_knn_evaluation(trainer, pl_module)

#     def run_knn_evaluation(self, trainer, pl_module):
#         pl_module.eval()  # Set model to eval mode
#         device = pl_module.device

#         # Use only a subset of the training dataloader
#         train_embeddings = []
#         train_labels = []
#         for i, t_batch in enumerate(self.train_dataloader):
#             if i >= self.max_train_batches:
#                 break
#             inputs, targets = t_batch
#             inputs = inputs.to(device)
#             with torch.no_grad():
#                 emb = pl_module(inputs)
#             train_embeddings.append(emb.cpu())
#             train_labels.append(targets.cpu())
#         X_train = torch.cat(train_embeddings, dim=0).numpy()
#         y_train = torch.cat(train_labels, dim=0).squeeze().numpy()

#         # Compute embeddings on the full validation set
#         val_embeddings = []
#         val_labels = []
#         for v_batch in self.val_dataloader:
#             inputs, targets = v_batch
#             inputs = inputs.to(device)
#             with torch.no_grad():
#                 emb = pl_module(inputs)
#             val_embeddings.append(emb.cpu())
#             val_labels.append(targets.cpu())
#         X_val = torch.cat(val_embeddings, dim=0).numpy()
#         y_val = torch.cat(val_labels, dim=0).squeeze().numpy()

#         # Fit a k-NN classifier on the training embeddings
#         knn = KNeighborsClassifier(n_neighbors=self.k)
#         knn.fit(X_train, y_train)
#         y_pred = knn.predict(X_val)

#         # Compute metrics
#         acc = accuracy_score(y_val, y_pred)
#         f1 = f1_score(y_val, y_pred, average='weighted')
#         try:
#             y_proba = knn.predict_proba(X_val)
#             if len(np.unique(y_val)) == 2:
#                 auc = roc_auc_score(y_val, y_proba[:, 1])
#             else:
#                 auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
#         except Exception:
#             auc = float('nan')

#         # Generate a confusion matrix plot
#         fig, ax = plt.subplots(figsize=(6, 6))
#         cm = confusion_matrix(y_val, y_pred)
#         cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
#         fig.colorbar(cax)
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=12)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')

#         # Generate t-SNE visualization of validation embeddings
#         from sklearn.manifold import TSNE
#         tsne = TSNE(n_components=2, random_state=42)
#         X_val_2d = tsne.fit_transform(X_val)
#         fig2, ax2 = plt.subplots(figsize=(8, 6))
#         scatter = ax2.scatter(X_val_2d[:, 0], X_val_2d[:, 1], c=y_val, cmap='viridis', alpha=0.7)
#         ax2.set_title("t-SNE Visualization of Validation Embeddings")
#         ax2.set_xlabel("t-SNE component 1")
#         ax2.set_ylabel("t-SNE component 2")
#         # Set discrete ticks based on the unique labels in y_val
#         classes = np.unique(y_val)
#         cbar = fig2.colorbar(scatter, ax=ax2, ticks=classes)
#         cbar.ax.set_yticklabels(classes)

#         # Log metrics and the confusion matrix & embeddings visualization images to WandB.
#         log_dict = {
#             "KNN Accuracy": acc,
#             "KNN F1 Score": f1,
#             "KNN AUC": auc,
#             "Confusion Matrix": wandb.Image(fig),
#             "Embeddings Visualization": wandb.Image(fig2)
#         }
#         trainer.logger.experiment.log(log_dict, step=trainer.global_step)
#         plt.close(fig)
#         plt.close(fig2)
#         pl_module.train()  # Switch back to training mode


if __name__ == "__main__":
    # Set your pretrained weights path
    pretrained_path = "/nfs/lambda_stor_01/homes/bhsu/2024_research/MedViT/weights/MedViT_base_im1k.pth"
    print(f'Reading in weights from here: {pretrained_path}')
    # Initialize the WandB logger
    wandb_logger = WandbLogger(project="MedViT_KNN_Eval")
    
    datasets = make_datasets()
    train_dataloader = datasets['train']
    val_dataloader = datasets['validation']
    n_classes = datasets['n_classes']
    # Initialize the Lightning module
    model = LitMedViT(n_classes=n_classes, lr=1e-4, pretrained_path=pretrained_path)
    
    # Create the k-NN evaluation callback.
    # Here we use a (larger-batch) training loader for evaluation and the test_loader for validation.
    knn_callback = KNN_Evaluation_Callback(train_dataloader=train_dataloader, 
                                           val_dataloader=val_dataloader, 
                                           k=5)
    # (Optional) Checkpoint callback to save model every 100 training steps.
    checkpoint_callback = ModelCheckpoint(
        dirpath="test_lightning_checkpoints",
        filename="{epoch}-{step}",
        every_n_train_steps=100 # every 100 batches
    )

    # Initialize the PyTorch Lightning trainer.
    # Here we select GPU index 1 (i.e. "cuda:1") if available.
    trainer = pl.Trainer(
        strategy='ddp',
        max_epochs=1,
        devices=[0, 1],
        logger=wandb_logger,
        callbacks=[knn_callback, checkpoint_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )
    # Start training.
    trainer.fit(model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader, 
                ckpt_path='/homes/bhsu/2024_research/MedViT/models_all_checkpoints/test_lightning_checkpoints/epoch=1-step=500.ckpt')
