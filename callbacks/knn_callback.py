import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import wandb
import pdb
from torch import Tensor, nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import umap.umap_ as umap
from typing import Tuple
from utils import BaseConfig


### tsne2 is a legacy util since shapes and colors too complex :(
def make_tsne2(X_val_2d: np.ndarray, y_val: np.ndarray, y_pred: np.ndarray) -> Figure: 
    """ Plot tsne and color by gold, shape by pred; I dont use this since its too hard to read :("""
    fig_tsne2, ax_tsne2 = plt.subplots(figsize=(8, 6))
    unique_true_labels = np.unique(y_val)
    unique_pred_labels = np.unique(y_pred)
    cmap = plt.get_cmap('viridis', len(unique_true_labels))
    label_to_color = {
        label: cmap(i) for i, label in enumerate(unique_true_labels)
    }
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'X']
    if len(unique_pred_labels) > len(markers):
        raise ValueError("Not enough markers for predicted labels. Expand marker list.")

    label_to_marker = {
        label: markers[i] for i, label in enumerate(unique_pred_labels)
    }

    for true_lb in unique_true_labels:
        for pred_lb in unique_pred_labels:
            mask = (y_val == true_lb) & (y_pred == pred_lb)
            ax_tsne2.scatter(
                X_val_2d[mask, 0], X_val_2d[mask, 1],
                color=label_to_color[true_lb],
                marker=label_to_marker[pred_lb],
                alpha=0.7,
                label=f"True={true_lb}, Pred={pred_lb}"
            )
    ax_tsne2.set_title("TSNE Plot of OCT Image Embeddings Colored by True Label and Shape by Predicted Label")
    ax_tsne2.set_xlabel("UMAP component 1")
    ax_tsne2.set_ylabel("UMAP component 2")

    color_handles = []
    for true_lb in unique_true_labels:
        color_patch = plt.Line2D(
            [0], [0],
            marker='o', color='w', label=str(true_lb),
            markerfacecolor=label_to_color[true_lb],
            markersize=8
        )
        color_handles.append(color_patch)
    marker_handles = []
    for pred_lb in unique_pred_labels:
        marker_patch = plt.Line2D(
            [0], [0],
            marker=label_to_marker[pred_lb], color='black',
            label=str(pred_lb), markerfacecolor='black',
            markersize=8
        )
        marker_handles.append(marker_patch)

    legend1 = ax_tsne2.legend(
        handles=color_handles, title="True Label", loc="upper left"
    )
    ax_tsne2.add_artist(legend1)
    legend2 = ax_tsne2.legend(
        handles=marker_handles, title="Pred Label", loc="lower left"
    )
    return fig_tsne2

####### actual code I use

class KNNCallBackConfig(BaseConfig): 
    k: int = 5
    """ k (int): Number of neighbors for k-NN."""
    log_every_n_steps: int = 100
    """ log_every_n_steps (int): Frequency of evaluation during training."""
    max_train_batches: int = 10
    """ 
    max_train_batches (int): Number of validation batches to use for evaluation so we dont compute for 
    all validation data
    """

def collect_embeddings(model: nn.Module, dataloader: DataLoader, num_batches: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Extract embeddings and labels """
    embeddings_list = []
    labels_list = []
    model.eval()
    inputs, targets = next(iter(dataloader))
    B = inputs.size()[0]
    with torch.no_grad():
        for idx, (inputs, targets) in tqdm(enumerate(dataloader),
                                    desc=f"Extracting Embeds Shape: {B}, Num Passes: {num_batches}",
                                    total=num_batches):
            inputs = inputs.to(model.device)
            if idx >= num_batches: 
                print(f'Collected {idx+1} train batches for KNN metrics callback')
                break
            emb = model(inputs)
            embeddings_list.append(emb.cpu())
            labels_list.append(targets.cpu())
    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).squeeze().numpy()
    return embeddings, labels

def make_roc_curve(true_values: np.ndarray, 
                   pred_values: np.ndarray, 
                   unique_labels: np.ndarray, 
                   auc_values: np.ndarray) -> Figure: 
    """ Returns multi-class ROC curve. I calculate per class and then one-vs-rest """
    fig_roc = None
    if pred_values is not None:
        fig_roc = plt.figure(figsize=(8, 6))
        ax_roc = fig_roc.add_subplot(111)
        if len(unique_labels) == 2:
            fpr, tpr, _ = roc_curve(true_values, pred_values[:, 1])                     # Binary classification
            ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc_values:.3f})")
            ax_roc.plot([0, 1], [0, 1], 'r--')
            ax_roc.set_title("ROC Curve (Binary)")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc='lower right')
        else:
            y_val_bin = label_binarize(true_values, classes=unique_labels)
            n_classes = y_val_bin.shape[1]
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], pred_values[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_val_bin.ravel(), pred_values.ravel()
            ) # Micro-average
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            for i in range(n_classes):
                ax_roc.plot(fpr[i], tpr[i],
                            label=f"Class {unique_labels[i]} (AUC={roc_auc[i]:.3f})")
            ax_roc.plot(fpr["micro"], tpr["micro"],
                        label=f"Micro-average (AUC={roc_auc['micro']:.3f})",
                        linestyle=':', linewidth=4)
            ax_roc.plot([0, 1], [0, 1], 'r--')
            ax_roc.set_title("ROC Curve (Multi-class: One-vs-Rest)")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc='lower right')
    return fig_roc

def make_confusion_matrix(predicted: np.ndarray, gold: np.ndarray) -> Figure:
    """ Returns a confusion matrix object """
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay.from_predictions(gold, predicted, cmap=plt.cm.Blues, ax=ax)
    disp.im_.set_clim(0, 250)  # Set the color limits of the image
    disp.ax_.set_xlabel('Predicted Labels')
    disp.ax_.set_ylabel('True Labels')
    disp.ax_.set_title('Confusion Matrix')
    return fig

def make_tsne(X_val_2d: np.ndarray, y_val: np.ndarray, split='train') -> Figure: 
    """ Returns a tsne plot """
    fig_tsne1, ax_tsne1 = plt.subplots(figsize=(8, 6))
    scatter1 = ax_tsne1.scatter(
        X_val_2d[:, 0], X_val_2d[:, 1],
        c=y_val, cmap='viridis', alpha=0.7
    )
    ax_tsne1.set_title(f"TSNE Plot of {split} Set OCT Embeddings\nColored by True Label")
    ax_tsne1.set_xlabel("TSNE Component 1")
    ax_tsne1.set_ylabel("TSNE Component 2")
    # Instead of a colorbar, create custom legend handles:
    classes = np.unique(y_val)
    # Create a normalizer and retrieve the colormap
    norm = plt.Normalize(vmin=classes.min(), vmax=classes.max())
    cmap = plt.cm.viridis
    # Build legend handles for each gold label
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=str(label),
                markerfacecolor=cmap(norm(label)), markersize=8)
        for label in classes
    ]
    # Add the legend to the plot
    ax_tsne1.legend(handles=legend_handles, title="True Label", loc='best')
    return fig_tsne1

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
        self.max_train_batches = max_train_batches 

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_count += 1
        if self.batch_count % self.log_every_n_steps == 0:
            print(f"Running Evaluation at global step {trainer.global_step}...")
            self.run_knn_evaluation(trainer, pl_module)

    def run_knn_evaluation(self, trainer, pl_module):
        """ Runs KNN evaluation by computing KNN + TSNE + other graphs used in section 5 """
        import pdb
        pl_module.eval() 
        device = pl_module.device
        
        # collect train and validation embeddings
        X_train, y_train = collect_embeddings(pl_module, self.train_dataloader, self.max_train_batches)
        X_val, y_val = collect_embeddings(pl_module, self.val_dataloader, self.max_train_batches)

        # Here is where I fit the knn and get the various accuracy metrics 
        print(f'KNN Training on X_train of Shape: {X_train.shape}')
        print(f'KNN Validating on X_val of Shape: {X_val.shape}')
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)
        
        # accuracy calculations
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        try:
            y_proba = knn.predict_proba(X_val)
            unique_labels = np.unique(y_val)
            if len(unique_labels) == 2:
                auc_val = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc_val = roc_auc_score(y_val, y_proba, multi_class='ovr')
        except Exception:
            y_proba = None
            auc_val = float('nan')
            
        fig_cm = make_confusion_matrix(y_pred, y_val)

        # tsne 
        tsne = TSNE(n_components=2, random_state=42)
        X_val_2d = tsne.fit_transform(X_val)
        print(f'TSNE Validating on X_val of Shape: {X_val.shape}')
        fig_tsne1 = make_tsne(X_val_2d, y_val, 'Validation')
        fig_tsne2 = make_tsne2(X_val_2d, y_val, y_pred)

        fig_roc = make_roc_curve(y_val, y_proba, unique_labels, auc_val)
        # I log to wandb
        log_dict = {
            "KNN Accuracy": acc,
            "KNN F1 Score": f1,
            "KNN AUC": auc_val,
            "Confusion Matrix of OCT Labels": wandb.Image(fig_cm),
            "t-SNE (Color by Label)": wandb.Image(fig_tsne1),
            "t-SNE (Color by Label, Shape by Pred)": wandb.Image(fig_tsne2),
        }
        if fig_roc is not None:
            log_dict["ROC Curve"] = wandb.Image(fig_roc)

        trainer.logger.experiment.log(log_dict, step=trainer.global_step)

        plt.close(fig_cm)
        plt.close(fig_tsne1)
        plt.close(fig_tsne2)
        if fig_roc is not None:
            plt.close(fig_roc)

        pl_module.train()
        
