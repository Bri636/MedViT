
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import wandb

from torch import Tensor, nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import umap.umap_ as umap
from typing import Tuple
from utils import BaseConfig

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
    
def collect_embeddings(model: nn.Module, dataloader: DataLoader, max_train_batches: int) -> Tuple[list[Tensor], list[Tensor]]: 
    """ Returns the embeddings with their labels as a tuple """
    embeddings = []
    labels = []
    with torch.no_grad(): 
        for idx, batch in dataloader: 
            inputs, targets = batch
            if idx >= max_train_batches: 
                print(f'Collected {idx+1} train batches for KNN metrics callback')
                break
            embedding= model(inputs)
            
            embeddings.append(embedding)
            labels.append(targets)
            
    X = torch.cat(embeddings, dim=0).numpy()
    y = torch.cat(labels, dim=0).squeeze().numpy() # dataloader labels have one extra dim
        
    return X, y


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
            # Multi-class: one-vs-rest approach
            # Binarize the labels for each class
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

def make_confusion_matrix(predicted: np.ndarray, gold: np.ndarray) -> Figure:
    """ Returns a confusion matrix object """ 
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
    cm = confusion_matrix(gold, predicted)
    cax = ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    cbar = fig_cm.colorbar(cax)
    # Explicitly add ticks at 0 and cm.max() so top tick is the largest:
    # cbar.set_ticks([0, cm.max()])
    # cbar.set_ticklabels([0, cm.max()])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(
                j, i, str(cm[i, j]),
                va='center', ha='center', fontsize=12
            )
    ax_cm.set_xlabel('Predicted Labels')
    ax_cm.set_ylabel('True Labels')
    ax_cm.set_title('Confusion Matrix')
    return fig_cm

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
        pl_module.eval()  # Set model to eval mode
        device = pl_module.device

        # ----------------------------
        # 1) Collect train embeddings
        # ----------------------------
        train_embeddings = []
        train_labels = []
        for i, t_batch in enumerate(self.train_dataloader):
            if i >= self.max_train_batches:
                print(f'Collected {i+1} train batches for KNN metrics callback')
                break
            inputs, targets = t_batch
            inputs = inputs.to(device)
            with torch.no_grad():
                emb = pl_module(inputs)
            train_embeddings.append(emb.cpu())
            train_labels.append(targets.cpu())

        X_train = torch.cat(train_embeddings, dim=0).numpy()
        y_train = torch.cat(train_labels, dim=0).squeeze().numpy()
    
        # ----------------------------
        # 2) Collect val embeddings
        # ----------------------------
        val_embeddings = []
        val_labels = []
        for j, v_batch in enumerate(self.val_dataloader):
            if j >= self.max_train_batches: # NOTE: max dataloader also
                print(f'Collected {j+1} train batches for KNN metrics callback')
                break
            inputs, targets = v_batch
            inputs = inputs.to(device)
            with torch.no_grad():
                emb = pl_module(inputs)
            val_embeddings.append(emb.cpu())
            val_labels.append(targets.cpu())

        X_val = torch.cat(val_embeddings, dim=0).numpy()
        y_val = torch.cat(val_labels, dim=0).squeeze().numpy()

        # ----------------------------
        # 3) Fit KNN and predict
        # ----------------------------
        print(f'KNN Training on X_train of Shape: {X_train.shape}')
        print(f'KNN Validating on X_val of Shape: {X_val.shape}')
        knn = KNeighborsClassifier(n_neighbors=self.k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        # ----------------------------
        # 4) Compute metrics
        # ----------------------------
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')

        # Probability outputs (if possible) for ROC/AUC
        try:
            y_proba = knn.predict_proba(X_val)
            unique_labels = np.unique(y_val)
            if len(unique_labels) == 2:
                # Binary
                auc_val = roc_auc_score(y_val, y_proba[:, 1])
            else:
                # Multi-class
                auc_val = roc_auc_score(y_val, y_proba, multi_class='ovr')
        except Exception:
            y_proba = None
            auc_val = float('nan')

        # ----------------------------
        # 5) Confusion Matrix
        # ----------------------------
        fig_cm, ax_cm = plt.subplots(figsize=(6, 6))
        cm = confusion_matrix(y_val, y_pred)
        cax = ax_cm.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
        cbar = fig_cm.colorbar(cax)
        # # Explicitly add ticks at 0 and cm.max() so top tick is the largest:
        # cbar.set_ticks([0, cm.max()])
        # cbar.set_ticklabels([0, cm.max()])

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(
                    j, i, str(cm[i, j]),
                    va='center', ha='center', fontsize=12
                )
        ax_cm.set_xlabel('Predicted Labels')
        ax_cm.set_ylabel('True Labels')
        ax_cm.set_title('Confusion Matrix')

        # ----------------------------
        # 6) t-SNE Visualization(s)
        # ----------------------------
        tsne = TSNE(n_components=2, random_state=42)
        X_val_2d = tsne.fit_transform(X_val)
        # NOTE: CHANGE from tsne
        print(f'TSNE Validating on X_val of Shape: {X_val.shape}')
        # umap_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
        # X_val_2d = umap_2d.fit_transform(X_val)

        # -- Plot A: color by gold label with a legend
        fig_tsne1, ax_tsne1 = plt.subplots(figsize=(8, 6))
        scatter1 = ax_tsne1.scatter(
            X_val_2d[:, 0], X_val_2d[:, 1],
            c=y_val, cmap='viridis', alpha=0.7
        )
        ax_tsne1.set_title("TSNE Plot of OCT Image Embeddings Colored by True Label")
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

        # -- Plot B: color by gold label, shape by predicted label
        fig_tsne2, ax_tsne2 = plt.subplots(figsize=(8, 6))

        # We need consistent color maps and marker types
        # Each true label => a color, each predicted label => a marker
        unique_true_labels = np.unique(y_val)
        unique_pred_labels = np.unique(y_pred)

        # Create color map for true labels
        # e.g. a dictionary: label -> color
        # Let's use a simple mapping using a standard colormap:
        cmap = plt.get_cmap('viridis', len(unique_true_labels))
        label_to_color = {
            label: cmap(i) for i, label in enumerate(unique_true_labels)
        }

        # Create a small list of markers to pick from for predicted labels
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'X']
        if len(unique_pred_labels) > len(markers):
            raise ValueError("Not enough markers for predicted labels. Expand marker list.")

        label_to_marker = {
            label: markers[i] for i, label in enumerate(unique_pred_labels)
        }

        # Plot data
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

        # Instead of a combined legend for all pairs, we can do two separate legends:
        #  - One for color
        #  - One for marker
        # We'll create custom handles for each

        # Color legend handles
        color_handles = []
        for true_lb in unique_true_labels:
            color_patch = plt.Line2D(
                [0], [0],
                marker='o', color='w', label=str(true_lb),
                markerfacecolor=label_to_color[true_lb],
                markersize=8
            )
            color_handles.append(color_patch)

        # Marker legend handles
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

        # ----------------------------
        # 7) ROC Curve
        # ----------------------------
        
        def make_roc_curve(true_values: np.ndarray, pred_values: np.ndarray) -> Figure: 
            """ Returns multi-class ROC curve """
            fig_roc = None
            if y_proba is not None:
                fig_roc = plt.figure(figsize=(8, 6))
                ax_roc = fig_roc.add_subplot(111)
                if len(unique_labels) == 2:
                    fpr, tpr, _ = roc_curve(y_val, y_proba[:, 1])                     # Binary classification
                    ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
                    ax_roc.plot([0, 1], [0, 1], 'r--')
                    ax_roc.set_title("ROC Curve (Binary)")
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.legend(loc='lower right')
                else:
                    # Multi-class: one-vs-rest approach
                    # Binarize the labels for each class
                    y_val_bin = label_binarize(y_val, classes=unique_labels)
                    n_classes = y_val_bin.shape[1]
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(n_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    fpr["micro"], tpr["micro"], _ = roc_curve(
                        y_val_bin.ravel(), y_proba.ravel()
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
            

        
        
        
        fig_roc = None
        if y_proba is not None:
            fig_roc = plt.figure(figsize=(8, 6))
            ax_roc = fig_roc.add_subplot(111)
            if len(unique_labels) == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_val, y_proba[:, 1])
                ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc_val:.3f})")
                ax_roc.plot([0, 1], [0, 1], 'r--')
                ax_roc.set_title("ROC Curve (Binary)")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend(loc='lower right')
            else:
                # Multi-class: one-vs-rest approach
                # Binarize the labels for each class
                y_val_bin = label_binarize(y_val, classes=unique_labels)
                n_classes = y_val_bin.shape[1]
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for i in range(n_classes):
                    fpr[i], tpr[i], _ = roc_curve(y_val_bin[:, i], y_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                
                # Micro-average
                fpr["micro"], tpr["micro"], _ = roc_curve(
                    y_val_bin.ravel(), y_proba.ravel()
                )
                roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

                # Plot each class
                for i in range(n_classes):
                    ax_roc.plot(fpr[i], tpr[i],
                                label=f"Class {unique_labels[i]} (AUC={roc_auc[i]:.3f})")
                # Plot micro-average
                ax_roc.plot(fpr["micro"], tpr["micro"],
                            label=f"Micro-average (AUC={roc_auc['micro']:.3f})",
                            linestyle=':', linewidth=4)

                ax_roc.plot([0, 1], [0, 1], 'r--')
                ax_roc.set_title("ROC Curve (Multi-class: One-vs-Rest)")
                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.legend(loc='lower right')
        # ----------------------------
        # 8) Log metrics and figures
        # ----------------------------
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

        # Close all figures
        plt.close(fig_cm)
        plt.close(fig_tsne1)
        plt.close(fig_tsne2)
        if fig_roc is not None:
            plt.close(fig_roc)

        # Switch back to training mode
        pl_module.train()
        
