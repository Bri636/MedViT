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

# ---------------------------
# Settings & Data Preparation
# ---------------------------
data_flag = 'octmnist'
download = True

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

print("Number of channels:", n_channels)
print("Number of classes:", n_classes)

# Use the same transformation for evaluation (avoid training augmentations)
eval_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda image: image.convert('RGB')),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load the train and test datasets.
train_dataset = DataClass(split='train', transform=eval_transform, download=download)
test_dataset = DataClass(split='test', transform=eval_transform, download=download)

# Create dataloaders. Feel free to increase batch_size if you have enough GPU memory.
BATCH_SIZE = 8
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Model Setup & Loading Checkpoint
# ---------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = MedViT_small(num_classes=n_classes).to(device)

# Load the checkpoint. Update the path below with your checkpoint file.
# checkpoint_path = '/homes/bhsu/2024_research/MedViT/checkpoints/epoch10_batch1500.pth'
checkpoint_path = '/homes/bhsu/2024_research/MedViT/og_lightning_checkpoints/epoch=9-step=15200.ckpt'
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict, strict=False)

# Bypass the classifier head so that forward() returns backbone embeddings.
model.proj_head = nn.Identity()

# Wrap the model’s forward pass to include L2 normalization.
def forward_embedding(x):
    emb = model(x)
    return nn.functional.normalize(emb, p=2, dim=1)

# Set the model in evaluation mode.
model.eval()

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

model_gradcam = copy.deepcopy(model)
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

def unnormalize(img_tensor):
    """
    Unnormalize a tensor image using mean=0.5 and std=0.5.
    Assumes img_tensor shape is (C, H, W).
    """
    return img_tensor * 0.5 + 0.5

def visualize_gradcam(image_tensor, heatmap, label, idx):
    """
    Displays the original image with the Grad-CAM heatmap overlay.
    """
    # Move tensor to CPU and unnormalize.
    image = unnormalize(image_tensor.cpu()).permute(1, 2, 0).numpy()
    # Squeeze heatmap to 2D.
    cam = heatmap.squeeze().cpu().numpy()
    
    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f"Image {idx} - Label: {label}")
    plt.axis('off')
    plt.savefig(f"Image-{idx}-Label-{label}.png")
    plt.show()

# Process one batch from the test loader.
# (Feel free to adjust the number of images visualized.)
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs = inputs.to(device)
    targets = targets.to(device)
    # Compute Grad-CAM heatmaps.
    # Here we let GradCAM choose the target class based on the prediction.
    cams = gradcam(inputs)  # Shape: [B, 1, H, W]
    
    # Visualize the first 5 images in the batch.
    for i in range(min(5, inputs.shape[0])):
        visualize_gradcam(inputs[i], cams[i], label=targets[i].item(), idx=i)
    break  # only process one batch

# Optionally, remove hooks when done.
gradcam.remove_hooks()