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

from pytorch_metric_learning import losses
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import medmnist
from medmnist import INFO
from MedViT import MedViT_small  # or MedViT_base, MedViT_large as needed
from utils import BaseConfig

class ModelConfig(BaseConfig): 
    ...

# ------------------------------
# Data Preparation
# ------------------------------

data_flag = 'octmnist'
download = True

# Hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
lr = 0.005

# Load medmnist info and dataset class
info = INFO[data_flag]
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

print("number of channels : ", n_channels)
print("number of classes : ", n_classes)

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda image: image.convert('RGB')),
    torchvision.transforms.AugMix(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# Create datasets and dataloaders
train_dataset = DataClass(split='train', transform=train_transform, download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# A separate loader for evaluation over the train set (using a larger batch size)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)

# ------------------------------
# Lightning Module Definition
# ------------------------------

class LitMedViT(pl.LightningModule):
    def __init__(self, n_classes, lr, pretrained_path):
        super().__init__()
        self.save_hyperparameters()  # logs hyperparameters to logger
        self.lr = lr
        self.n_classes = n_classes
        
        # Initialize the MedViT model and load pretrained weights
        self.model = MedViT_small(num_classes=n_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(pretrained_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=False)
        # Bypass the classifier head so that forward returns backbone embeddings.
        self.model.proj_head = nn.Identity()
        
        # Contrastive loss (SupConLoss)
        self.metric_loss_func = losses.SupConLoss(temperature=0.07)
        
    def forward(self, x):
        # Forward pass returns L2 normalized embeddings
        emb = self.model(x)
        return F.normalize(emb, p=2, dim=1)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        # Compute embeddings
        embeddings = self.forward(inputs)
        # Ensure targets are 1D (in case they come in shape [B, 1])
        loss = self.metric_loss_func(embeddings, targets.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # In this example, validation_step does not compute a loss since k-NN evaluation is done in the callback.
        inputs, targets = batch
        embeddings = self.forward(inputs)
        return {"embeddings": embeddings.cpu(), "targets": targets.cpu()}
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        return optimizer

# ------------------------------
# Custom Callback for k-NN Evaluation & Logging to WandB
# ------------------------------

# class KNN_Evaluation_Callback(Callback):
#     def __init__(self, train_dataloader, val_dataloader, k=5):
#         super().__init__()
#         self.train_dataloader = train_dataloader
#         self.val_dataloader = val_dataloader
#         self.k = k
    
#     def on_validation_epoch_end(self, trainer, pl_module):
#         pl_module.eval()
#         device = pl_module.device
        
#         # Compute embeddings on the (subset of) training set
#         train_embeddings = []
#         train_labels = []
#         for batch in self.train_dataloader:
#             inputs, targets = batch
#             inputs = inputs.to(device)
#             with torch.no_grad():
#                 emb = pl_module(inputs)
#             train_embeddings.append(emb.cpu())
#             train_labels.append(targets.cpu())
#         X_train = torch.cat(train_embeddings, dim=0).numpy()
#         y_train = torch.cat(train_labels, dim=0).squeeze().numpy()
        
#         # Compute embeddings on the validation set
#         val_embeddings = []
#         val_labels = []
#         for batch in self.val_dataloader:
#             inputs, targets = batch
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
#             # For binary classification use the probability for the positive class;
#             # otherwise, use one-vs-rest
#             if len(np.unique(y_val)) == 2:
#                 auc = roc_auc_score(y_val, y_proba[:, 1])
#             else:
#                 auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
#         except Exception as e:
#             auc = float('nan')
        
#         # Generate a confusion matrix plot
#         cm = confusion_matrix(y_val, y_pred)
#         fig, ax = plt.subplots(figsize=(6, 6))
#         cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
#         fig.colorbar(cax)
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', fontsize=12)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')
        
#         # Log metrics and the confusion matrix image to WandB.
#         log_dict = {
#             "KNN Accuracy": acc,
#             "KNN F1 Score": f1,
#             "KNN AUC": auc,
#             "Confusion Matrix": wandb.Image(fig)
#         }
#         # Log using the experiment associated with the WandbLogger.
#         trainer.logger.experiment.log(log_dict, step=trainer.global_step)
#         plt.close(fig)
#         pl_module.train()

# class KNN_Evaluation_Callback(Callback):
#     def __init__(self, train_dataloader, val_dataloader, k=5, log_every_n_steps=100, max_train_batches=10):
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
#         cm = confusion_matrix(y_val, y_pred)
#         fig, ax = plt.subplots(figsize=(6, 6))
#         cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
#         fig.colorbar(cax)
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=12)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')

#         # Log metrics and the confusion matrix image to WandB.
#         log_dict = {
#             "KNN Accuracy": acc,
#             "KNN F1 Score": f1,
#             "KNN AUC": auc,
#             "Confusion Matrix": wandb.Image(fig)
#         }
#         trainer.logger.experiment.log(log_dict, step=trainer.global_step)
#         plt.close(fig)
#         pl_module.train()  # Switch back to training mode


# class KNN_Evaluation_Callback(Callback):
#     def __init__(self, train_dataloader, val_dataloader, k=5, log_every_n_steps=100, max_train_batches=10):
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
#         cm = confusion_matrix(y_val, y_pred)
#         fig, ax = plt.subplots(figsize=(6, 6))
#         cax = ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
#         fig.colorbar(cax)
#         for i in range(cm.shape[0]):
#             for j in range(cm.shape[1]):
#                 ax.text(j, i, str(cm[i, j]), va='center', ha='center', fontsize=12)
#         plt.xlabel('Predicted')
#         plt.ylabel('True')
#         plt.title('Confusion Matrix')

#         # Generate TSNE visualization of validation embeddings
#         from sklearn.manifold import TSNE
#         tsne = TSNE(n_components=2, random_state=42)
#         X_val_2d = tsne.fit_transform(X_val)
#         fig2, ax2 = plt.subplots(figsize=(8, 6))
#         scatter = ax2.scatter(X_val_2d[:, 0], X_val_2d[:, 1], c=y_val, cmap='viridis', alpha=0.7)
#         ax2.set_title("t-SNE Visualization of Validation Embeddings")
#         ax2.set_xlabel("t-SNE component 1")
#         ax2.set_ylabel("t-SNE component 2")
#         fig2.colorbar(scatter, ax=ax2)

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


class KNN_Evaluation_Callback(Callback):
    def __init__(self, train_dataloader, val_dataloader, k=5, log_every_n_steps=100, max_train_batches=10):
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




# ------------------------------
# Main Training Script
# ------------------------------

if __name__ == "__main__":
    # Set your pretrained weights path
    pretrained_path = "./weights/MedViT_base_im1k.pth"
    print(f'Reading in weights from here: {pretrained_path}')
    # Initialize the WandB logger
    wandb_logger = WandbLogger(project="MedViT_KNN_Eval")
    
    # Initialize the Lightning module
    model = LitMedViT(n_classes=n_classes, lr=lr, pretrained_path=pretrained_path)
    
    # Create the k-NN evaluation callback.
    # Here we use a (larger-batch) training loader for evaluation and the test_loader for validation.
    knn_callback = KNN_Evaluation_Callback(train_dataloader=train_loader_at_eval, val_dataloader=test_loader, k=5)
    
    # (Optional) Checkpoint callback to save model every 100 training steps.
    checkpoint_callback = ModelCheckpoint(
        dirpath="lightning_checkpoints",
        filename="{epoch}-{step}",
        every_n_train_steps=100
    )
    
    # Initialize the PyTorch Lightning trainer.
    # Here we select GPU index 1 (i.e. "cuda:1") if available.
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        devices=[0],
        logger=wandb_logger,
        callbacks=[knn_callback, checkpoint_callback],
        log_every_n_steps=10,
        num_sanity_val_steps=0
    )
    
    # Start training.
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)
