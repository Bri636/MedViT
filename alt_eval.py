import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
import medmnist
from medmnist import INFO
from models.MedViT import MedViT_small  # or MedViT_base, MedViT_large as required
from tqdm import tqdm
import copy
from utils import GradCAM
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import Tuple
from pathlib import Path
import itertools
# import packages
from medmnist_datasets.medmnist_dataset import make_dataloaders, _DATA_FLAGS
from utils import BaseConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURRENT_DIR = Path(__file__).resolve().parent
MODEL_CKPT_PATH = f'{CURRENT_DIR}/models_all_checkpoints/good_checkpoints/epoch10_batch1500.pth'
IMG_SAVE_PATH = f'{CURRENT_DIR}/images/'

def forward_embedding(x: torch.Tensor) -> torch.Tensor:
    """ l2 normalize """
    return nn.functional.normalize(x, p=2, dim=1)

def make_gradcam(model_gradcam: torch.nn.Module) -> GradCAM: 
    """ make gradcam object for evaluting my model  """
    model_gradcam.eval()
    # first convolutional layer from the patch embedding of the first block. embed has best representations
    if hasattr(model_gradcam, 'features') and hasattr(model_gradcam.features[0], 'patch_embed'):
        # Use the convolutional layer from the patch embedding of the first block.
        gradcam_layer = model_gradcam.features[0].patch_embed.conv
    return GradCAM(model_gradcam, gradcam_layer)

def unnormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    """
    Unnormalize a tensor image using mean=0.5 and std=0.5.
    Assumes img_tensor shape is (C, H, W).
    """
    return img_tensor * 0.5 + 0.5

def get_embeddings(dataloader: data.DataLoader, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Extract embeddings and labels """
    embeddings_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(DEVICE)
            emb = model(inputs)
            embeddings_list.append(emb.cpu())
            labels_list.append(targets.cpu())
    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    labels = torch.cat(labels_list, dim=0).squeeze().numpy()
    return embeddings, labels

def visualize_gradcam(image_tensor: torch.Tensor, 
                      heatmap: torch.Tensor, 
                      estimated_label: int, 
                      gold_label: int, 
                      idx: int) -> None:
    """
    Displays a side-by-side figure with the original image and the Grad-CAM heatmap overlay.
    The figure title includes 'Estimated Label' (model prediction) and 'Gold Label' (true label).
    """
    image = unnormalize(image_tensor).permute(1, 2, 0).cpu().numpy()
    cam = heatmap.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    axes[1].imshow(image)
    axes[1].imshow(cam, cmap='jet', alpha=0.5)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis('off')
    
    fig.suptitle(f"Estimated Label: {estimated_label}, Gold Label: {gold_label}")
    plt.savefig(f"GradCAM-Image-{idx}-Gold-{gold_label}-Est-{estimated_label}.png")
    plt.show()

def parse_arguments(): 
    argparser = ArgumentParser()
    argparser.add_argument('--model_checkpoint_path', 
                           type=str, 
                           default=MODEL_CKPT_PATH)
    argparser.add_argument('--batch_size', type=int, default=24)
    argparser.add_argument('--images_save_path', 
                           type=str, 
                           default=IMG_SAVE_PATH)
    return argparser.parse_args()

def main(): 
    args = parse_arguments()
    dataloaders = make_dataloaders('octmnist', args.batch_size)
    train_dataloader = dataloaders['train']
    small_num_batches = int(len(train_dataloader) * 0.2)
    train_dataloader = itertools.islice(train_dataloader, small_num_batches)
    test_dataloader = dataloaders['test']
    n_classes = dataloaders['n_classes']
    model = MedViT_small(num_classes=n_classes).to(DEVICE)
    state_dict = torch.load(args.model_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.proj_head = nn.Identity()
    model.eval()
    
    print("Extracting train embeddings for k-NN classifier...")
    train_embeddings, train_labels = get_embeddings(train_dataloader, model)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings, train_labels)
    
    gradcam = make_gradcam(copy.deepcopy(model))
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        with torch.no_grad():
            test_embeddings = model(inputs).cpu().numpy()
        knn_predictions = knn.predict(test_embeddings)
        cams = gradcam(inputs)  # Shape: [B, 1, H, W]
        for i in tqdm(range(min(args.batch_size, inputs.shape[0])), desc='Generating Images'):
            estimated_label = knn_predictions[i]
            gold_label = targets[i].item()
            visualize_gradcam(inputs[i], cams[i], estimated_label, gold_label, idx=i)
        break  # only process one batch
    gradcam.remove_hooks()
    
    targets_np = targets.cpu().numpy()
    accuracy = accuracy_score(targets_np, knn_predictions)
    f1 = f1_score(targets_np, knn_predictions, average='weighted')
    prob_estimates = knn.predict_proba(test_embeddings)
    if n_classes == 2:
        auc = roc_auc_score(targets_np, prob_estimates[:, 1])
    else:
        auc = roc_auc_score(targets_np, prob_estimates, multi_class='ovr')
        
    print("===================================")
    print("k-NN Evaluation on Test Set:")
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    print("===================================")
    print("\nClassification Report:")
    print(classification_report(targets_np, knn_predictions))
        
if __name__=="__main__": 
    main()