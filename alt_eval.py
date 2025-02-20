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

# import packages
from medmnist_datasets.medmnist_dataset import make_datasets, _DATA_FLAGS
from utils import BaseConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# datasets = make_datasets(DATA_FLAG, BATCH_SIZE)
# test_dataloader = datasets['test']
# n_classes = datasets['n_classes']

# model = MedViT_small(num_classes=n_classes).to(device)

# # Load the checkpoint. Update the path below with your checkpoint file.
# # checkpoint_path = '/homes/bhsu/2024_research/MedViT/checkpoints/epoch10_batch1500.pth'
# checkpoint_path = '/homes/bhsu/2024_research/MedViT/og_lightning_checkpoints/epoch=9-step=15200.ckpt'
# state_dict = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(state_dict, strict=False)
# model.proj_head = nn.Identity()
# model.eval()

# # ---------------------------
# # Extract Embeddings from Data
# # ---------------------------
# def get_embeddings(dataloader):
#     embeddings_list = []
#     labels_list = []
#     with torch.no_grad():
#         for inputs, targets in tqdm(dataloader):
#             inputs = inputs.to(device)
#             emb = forward_embedding(inputs)
#             embeddings_list.append(emb.cpu())
#             labels_list.append(targets.cpu())
#     embeddings = torch.cat(embeddings_list, dim=0).numpy()
#     # Ensure labels are 1D (squeeze in case they are of shape [N,1])
#     labels = torch.cat(labels_list, dim=0).squeeze().numpy()
#     return embeddings, labels

# print("Extracting train embeddings...")
# train_embeddings, train_labels = get_embeddings(train_loader)
# print("Extracting test embeddings...")
# test_embeddings, test_labels = get_embeddings(test_loader)

# # ---------------------------
# # k-NN Classification & Evaluation
# # ---------------------------
# # Create and train the k-NN classifier (here using k=5, but you can modify as needed)
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(train_embeddings, train_labels)

# # Predict labels for the test set.
# predictions = knn.predict(test_embeddings)

# # Calculate evaluation metrics.
# accuracy = accuracy_score(test_labels, predictions)
# f1 = f1_score(test_labels, predictions, average='weighted')

# # For AUC, we need prediction probabilities.
# # For binary classification, use the probability for the positive class.
# # For multi-class, use the one-vs-rest scheme.
# probas = knn.predict_proba(test_embeddings)
# if n_classes == 2:
#     auc = roc_auc_score(test_labels, probas[:, 1])
# else:
#     auc = roc_auc_score(test_labels, probas, multi_class='ovr')

# print("===================================")
# print("k-NN Evaluation on Test Set:")
# print(f"Accuracy: {accuracy*100:.2f}%")
# print(f"F1 Score: {f1:.4f}")
# print(f"AUC: {auc:.4f}")
# print("===================================")
# print("\nClassification Report:")
# print(classification_report(test_labels, predictions))


class EvalConfig(BaseConfig): 
    model_checkpoint_path: str = '/homes/bhsu/2024_research/MedViT/models_all_checkpoints/good_checkpoints/epoch10_batch1500.pth'
    data_flag: _DATA_FLAGS = 'octmnist'
    batch_size: int = 32

def forward_embedding(x: torch.Tensor) -> torch.Tensor:
    """ l2 normalize """
    return nn.functional.normalize(x, p=2, dim=1)

def make_gradcam(model_gradcam: torch.nn.Module) -> GradCAM: 
    """ deepcopies model to initialize gradcam for eval """
    model_gradcam.eval()
    # (Update the target layer as needed based on your MedViT implementation)
    # Determine Target Layer for Grad-CAM for MedViT
    if hasattr(model_gradcam, 'features') and hasattr(model_gradcam.features[0], 'patch_embed'):
        # Use the convolutional layer from the patch embedding of the first block.
        target_layer = model_gradcam.features[0].patch_embed.conv
    else:
        raise ValueError("Target layer for GradCAM not found. Please update the target layer accordingly.")
    # Initialize GradCAM with the gradcam model and target layer.
    gradcam = GradCAM(model_gradcam, target_layer)
    return gradcam

def unnormalize(img_tensor):
    """
    Unnormalize a tensor image using mean=0.5 and std=0.5.
    Assumes img_tensor shape is (C, H, W).
    """
    return img_tensor * 0.5 + 0.5

# def visualize_gradcam(image_tensor, heatmap, label, idx):
#     """
#     Displays the original image with the Grad-CAM heatmap overlay.
#     """
#     # Move tensor to CPU and unnormalize.
#     image = unnormalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
#     # Squeeze heatmap to 2D.
#     cam = heatmap.squeeze().cpu().numpy()
    
#     plt.figure(figsize=(6,6))
#     plt.imshow(image)
#     plt.imshow(cam, cmap='jet', alpha=0.5)
#     plt.title(f"Image {idx} - Label: {label}")
#     plt.axis('off')
#     plt.savefig(f"Image-{idx}-Label-{label}.png")
#     plt.show()
    
# def compute_saliency_maps(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor = None, normalize: bool = True) -> torch.Tensor:
#     """
#     Computes saliency maps for a batch of inputs.
    
#     Args:
#         model: The neural network model.
#         inputs: A batch of input images (shape: [B, C, H, W]).
#         targets: (Optional) The target class indices. If None, uses the predicted class.
#         normalize: If True, normalize each saliency map to [0,1].
    
#     Returns:
#         A tensor of saliency maps with shape [B, H, W].
#     """
#     model.eval()
#     inputs = inputs.clone().detach()
#     inputs.requires_grad_()  # Enable gradients for the input

#     # Forward pass
#     outputs = model(inputs)  # outputs shape: [B, num_classes]
    
#     # If targets are not provided, use the predicted class (argmax) for each example.
#     if targets is None:
#         targets = outputs.argmax(dim=1)
    
#     # Select the scores for the target classes.
#     # This gathers the output for each image corresponding to its target class.
#     target_scores = outputs.gather(1, targets.view(-1, 1)).squeeze()
    
#     # Sum up the target scores and compute gradients.
#     loss = target_scores.sum()
#     model.zero_grad()
#     loss.backward()
    
#     # Get the absolute gradients with respect to the input.
#     saliency = inputs.grad.data.abs()  # shape: [B, C, H, W]
    
#     # For each pixel, take the maximum across the color channels.
#     saliency, _ = saliency.max(dim=1)  # shape: [B, H, W]
    
#     if normalize:
#         # Normalize each saliency map to [0, 1]
#         saliency_min = saliency.view(saliency.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
#         saliency_max = saliency.view(saliency.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
#         saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
    
#     return saliency

# def visualize_saliency(image_tensor: torch.Tensor, saliency_map: torch.Tensor, label: int, idx: int):
#     """
#     Overlays the saliency map on the original image and saves the visualization.
    
#     Args:
#         image_tensor: The original image tensor (C, H, W). Assumes it is normalized.
#         saliency_map: A 2D saliency map (H, W) corresponding to the image.
#         label: The label of the image.
#         idx: An index used for saving the file.
#     """
#     # Unnormalize image (assuming mean=0.5, std=0.5)
#     image = unnormalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
    
#     # Convert saliency map to numpy.
#     saliency = saliency_map.cpu().numpy()
    
#     plt.figure(figsize=(6, 6))
#     plt.imshow(image)
#     plt.imshow(saliency, cmap='hot', alpha=0.5)
#     plt.title(f"Image {idx} - Label: {label}")
#     plt.axis('off')
#     # Save the visualization
#     save_path = f"Saliency-Image-{idx}-Label-{label}.png"
#     plt.savefig(save_path)
#     plt.show()


def get_embeddings(dataloader, model):
    """
    Extract embeddings and labels from the dataloader using the provided model.
    """
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

def visualize_gradcam(image_tensor, heatmap, estimated_label, gold_label, idx):
    """
    Displays a side-by-side figure with the original image and the Grad-CAM heatmap overlay.
    The figure title includes 'Estimated Label' (model prediction) and 'Gold Label' (true label).
    """
    # Move tensor to CPU and unnormalize.
    image = unnormalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
    # Squeeze heatmap to 2D.
    cam = heatmap.squeeze().cpu().numpy()
    
    # Create a side-by-side plot.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Left: original image.
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    # Right: overlay heatmap.
    axes[1].imshow(image)
    axes[1].imshow(cam, cmap='jet', alpha=0.5)
    axes[1].set_title("Grad-CAM Overlay")
    axes[1].axis('off')
    
    # Set the overall title.
    fig.suptitle(f"Estimated Label: {estimated_label}, Gold Label: {gold_label}")
    plt.savefig(f"GradCAM-Image-{idx}-Gold-{gold_label}-Est-{estimated_label}.png")
    plt.show()

def compute_saliency_maps(model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor = None, normalize: bool = True) -> torch.Tensor:
    """
    Computes saliency maps for a batch of inputs.
    
    Args:
        model: The neural network model.
        inputs: A batch of input images (shape: [B, C, H, W]).
        targets: (Optional) The target class indices. If None, uses the predicted class.
        normalize: If True, normalize each saliency map to [0,1].
    
    Returns:
        A tensor of saliency maps with shape [B, H, W].
    """
    model.eval()
    inputs = inputs.clone().detach()
    inputs.requires_grad_()  # Enable gradients for the input

    # Forward pass
    outputs = model(inputs)  # outputs shape: [B, num_classes]
    
    # If targets are not provided, use the predicted class (argmax) for each example.
    if targets is None:
        targets = outputs.argmax(dim=1)
    
    # Select the scores for the target classes.
    target_scores = outputs.gather(1, targets.view(-1, 1)).squeeze()
    
    # Sum up the target scores and compute gradients.
    loss = target_scores.sum()
    model.zero_grad()
    loss.backward()
    
    # Get the absolute gradients with respect to the input.
    saliency = inputs.grad.data.abs()  # shape: [B, C, H, W]
    
    # For each pixel, take the maximum across the color channels.
    saliency, _ = saliency.max(dim=1)  # shape: [B, H, W]
    
    if normalize:
        # Normalize each saliency map to [0, 1]
        saliency_min = saliency.view(saliency.size(0), -1).min(dim=1)[0].view(-1, 1, 1)
        saliency_max = saliency.view(saliency.size(0), -1).max(dim=1)[0].view(-1, 1, 1)
        saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)
    
    return saliency

def visualize_saliency(image_tensor: torch.Tensor, saliency_map: torch.Tensor, estimated_label, gold_label, idx: int):
    """
    Displays a side-by-side figure with the original image and the saliency map overlay.
    The figure title includes 'Estimated Label' (model prediction) and 'Gold Label' (true label).
    
    Args:
        image_tensor: The original image tensor (C, H, W). Assumes it is normalized.
        saliency_map: A 2D saliency map (H, W) corresponding to the image.
        estimated_label: The predicted label for the image.
        gold_label: The true label of the image.
        idx: An index used for saving the file.
    """
    # Unnormalize image (assuming mean=0.5, std=0.5)
    image = unnormalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
    
    # Convert saliency map to numpy.
    saliency = saliency_map.cpu().numpy()
    
    # Create a side-by-side plot.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Left: original image.
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    # Right: overlay saliency map.
    axes[1].imshow(image)
    axes[1].imshow(saliency, cmap='hot', alpha=0.5)
    axes[1].set_title("Saliency Map Overlay")
    axes[1].axis('off')
    
    # Set the overall title.
    fig.suptitle(f"Estimated Label: {estimated_label}, Gold Label: {gold_label}")
    plt.savefig(f"Saliency-Image-{idx}-Gold-{gold_label}-Est-{estimated_label}.png")
    plt.show()



def parse_arguments(): 
    argparser = ArgumentParser()
    argparser.add_argument('--model_checkpoint_path', 
                           type=str, 
                           default='/homes/bhsu/2024_research/MedViT/models_all_checkpoints/good_checkpoints/epoch10_batch1500.pth')
    argparser.add_argument('--eval_config', type=str)
    return argparser.parse_args()

def main(): 
    args = parse_arguments()
    # eval_config = EvalConfig.from_yaml(args.eval_config)
    batch_size = 8
    datasets = make_datasets('octmnist', batch_size)
    test_dataloader = datasets['test']
    n_classes = datasets['n_classes']

    model = MedViT_small(num_classes=n_classes).to(DEVICE)
    state_dict = torch.load(args.model_checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.proj_head = nn.Identity()
    model.eval()
    
    print("Extracting train embeddings for k-NN classifier...")
    train_embeddings, train_labels = get_embeddings(test_dataloader, model)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings, train_labels)
    
    gradcam = make_gradcam(copy.deepcopy(model))
    for batch_idx, (inputs, targets) in enumerate(test_dataloader):
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        
        # Compute test embeddings and get estimated labels from the k-NN classifier.
        with torch.no_grad():
            test_embeddings = model(inputs).cpu().numpy()
        knn_predictions = knn.predict(test_embeddings)
        # breakpoint()
        # Compute Grad-CAM heatmaps.
        # Here we let GradCAM choose the target class based on the prediction.
        cams = gradcam(inputs)  # Shape: [B, 1, H, W]
        sal_maps = compute_saliency_maps(model, inputs, targets)
        # Visualize the first 5 images in the batch.
        for i in range(min(batch_size, inputs.shape[0])):
            # Compute the estimated label using the model prediction.
            estimated_label = knn_predictions[i]
            gold_label = targets[i].item()
            # visualize_gradcam(inputs[i], cams[i], label=targets[i].item(), idx=i)
            # visualize_saliency(inputs[i], sal_maps[i], label=targets[i].item(), idx=i)
            visualize_gradcam(inputs[i], cams[i], estimated_label, gold_label, idx=i)
            visualize_saliency(inputs[i], sal_maps[i], estimated_label, gold_label, idx=i)
        break  # only process one batch
    gradcam.remove_hooks()
        
if __name__=="__main__": 
    
    main()