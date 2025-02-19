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
# from callbacks.knn_callback import KNNCallBackConfig, KNN_Evaluation_Callback
# from medmnist_datasets.medmnist_dataset import make_datasets
from models.MedViT import MedViT_small  # or MedViT_base, MedViT_large as needed
# from utils import BaseConfig

# # some defaults yeh
# NUM_EPOCHS = 10
# LR = 0.005

# class TrainConfig(BaseConfig): 
#     """ Training config """
#     model_checkpoint_path: str 
#     num_epochs: int = 10 
#     lr: float = 0.005

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
        # I swap classifier head; forward returns backbone embeddings.
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

# if __name__ == "__main__":
#     # Set your pretrained weights path
#     pretrained_path = "/nfs/lambda_stor_01/homes/bhsu/2024_research/MedViT/weights/MedViT_base_im1k.pth"
#     print(f'Reading in weights from here: {pretrained_path}')
#     # Initialize the WandB logger
#     wandb_logger = WandbLogger(project="MedViT_KNN_Eval")
    
#     datasets = make_datasets()
#     train_dataloader = datasets['train']
#     val_dataloader = datasets['validation']
#     n_classes = datasets['n_classes']
#     # Initialize the Lightning module
#     model = LitMedViT(n_classes=n_classes, lr=1e-4, pretrained_path=pretrained_path)
    
#     # Create the k-NN evaluation callback.
#     # Here we use a (larger-batch) training loader for evaluation and the test_loader for validation.
#     knn_callback = KNN_Evaluation_Callback(train_dataloader=train_dataloader, 
#                                            val_dataloader=val_dataloader, 
#                                            k=5)
#     # (Optional) Checkpoint callback to save model every 100 training steps.
#     checkpoint_callback = ModelCheckpoint(
#         dirpath="test_lightning_checkpoints",
#         filename="{epoch}-{step}",
#         every_n_train_steps=100 # every 100 batches
#     )

#     # Initialize the PyTorch Lightning trainer.
#     # Here we select GPU index 1 (i.e. "cuda:1") if available.
#     trainer = pl.Trainer(
#         strategy='ddp',
#         max_epochs=1,
#         devices=[0, 1],
#         logger=wandb_logger,
#         callbacks=[knn_callback, checkpoint_callback],
#         log_every_n_steps=10,
#         num_sanity_val_steps=0
#     )
#     # Start training.
#     trainer.fit(model, 
#                 train_dataloaders=train_dataloader, 
#                 val_dataloaders=val_dataloader, 
#                 ckpt_path='/homes/bhsu/2024_research/MedViT/models_all_checkpoints/test_lightning_checkpoints/epoch=1-step=500.ckpt')
