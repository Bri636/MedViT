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
from MedViT import MedViT_small  # or MedViT_base, MedViT_large as required
from tqdm import tqdm
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
BATCH_SIZE = 128
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------
# Model Setup & Loading Checkpoint
# ---------------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model = MedViT_small(num_classes=n_classes).to(device)

# Load the checkpoint. Update the path below with your checkpoint file.
checkpoint_path = '/homes/bhsu/2024_research/MedViT/checkpoints/epoch10_batch1500.pth'
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

# ---------------------------
# Extract Embeddings from Data
# ---------------------------
def get_embeddings(dataloader):
    embeddings_list = []
    labels_list = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.to(device)
            emb = forward_embedding(inputs)
            embeddings_list.append(emb.cpu())
            labels_list.append(targets.cpu())
    embeddings = torch.cat(embeddings_list, dim=0).numpy()
    # Ensure labels are 1D (squeeze in case they are of shape [N,1])
    labels = torch.cat(labels_list, dim=0).squeeze().numpy()
    return embeddings, labels

print("Extracting train embeddings...")
train_embeddings, train_labels = get_embeddings(train_loader)
print("Extracting test embeddings...")
test_embeddings, test_labels = get_embeddings(test_loader)

# ---------------------------
# k-NN Classification & Evaluation
# ---------------------------
# Create and train the k-NN classifier (here using k=5, but you can modify as needed)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings, train_labels)

# Predict labels for the test set.
predictions = knn.predict(test_embeddings)

# Calculate evaluation metrics.
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')

# For AUC, we need prediction probabilities.
# For binary classification, use the probability for the positive class.
# For multi-class, use the one-vs-rest scheme.
probas = knn.predict_proba(test_embeddings)
if n_classes == 2:
    auc = roc_auc_score(test_labels, probas[:, 1])
else:
    auc = roc_auc_score(test_labels, probas, multi_class='ovr')

print("===================================")
print("k-NN Evaluation on Test Set:")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"F1 Score: {f1:.4f}")
print(f"AUC: {auc:.4f}")
print("===================================")
print("\nClassification Report:")
print(classification_report(test_labels, predictions))
