import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_metric_learning import losses  # using SupConLoss for contrastive training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from MedViT import MedViT_small

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary

from tqdm import tqdm
import medmnist
from medmnist import INFO, Evaluator

import torchattacks
from torchattacks import PGD, FGSM
from MedViT import MedViT_small, MedViT_base, MedViT_large

torch.cuda.set_device(1)

data_flag = 'octmnist'
# [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
# pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist]
download = True

NUM_EPOCHS = 10
BATCH_SIZE = 64
lr = 0.005

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

print("number of channels : ", n_channels)
print("number of classes : ", n_classes)

from torchvision.transforms.transforms import Resize
# preprocessing
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

# load the data
train_dataset = DataClass(split='train', transform=train_transform, download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)

# pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)

device = 'cuda:1'

# ----- Model Initialization & Pretrained Weights Loading -----
model = MedViT_small(num_classes=n_classes).cuda()
pretrained_path = '/homes/bhsu/2024_research/med_class/MedViT/weights/MedViT_base_im1k.pth'
state_dict = torch.load(pretrained_path, map_location=device)
model.load_state_dict(state_dict, strict=False)

# Bypass the classifier head so that forward returns backbone embeddings.
model.proj_head = nn.Identity()

# Optional: wrap forward to include L2 normalization.
def forward_embedding(x):
    emb = model(x)
    return nn.functional.normalize(emb, p=2, dim=1)

# ----- Optimizer & Contrastive Loss Setup -----
metric_loss_func = losses.SupConLoss(temperature=0.07)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

checkpoint_dir = "alt_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_interval = 100  # save checkpoint every 10 batches

# ----- Training Loop with Periodic Classification Evaluation -----
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    embeddings_list = []
    labels_list = []
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        embeddings = forward_embedding(inputs)
        loss = metric_loss_func(embeddings, targets.squeeze(1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Accumulate embeddings and targets for periodic evaluation
        embeddings_list.append(embeddings.detach().cpu())
        labels_list.append(targets.detach().cpu())
        
        if (batch_idx + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch{epoch+1}_batch{batch_idx+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            print(f"Batch Loss: {running_loss / (batch_idx + 1):.4f}")
            
            # --- Optional Classification Evaluation ---
            # Concatenate accumulated embeddings and labels.
            X_train = torch.cat(embeddings_list, dim=0).numpy()
            y_train = torch.cat(labels_list, dim=0).numpy()
            
            # Train a k-NN classifier (for example, k=5) on the current batch's embeddings.
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train, y_train)
            
            # Evaluate on the validation set.
            model.eval()
            val_embeddings = []
            val_labels = []
            with torch.no_grad():
                for val_inputs, val_targets in test_loader:
                    val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()
                    emb = forward_embedding(val_inputs)
                    val_embeddings.append(emb.detach().cpu())
                    val_labels.append(val_targets.detach().cpu())
            X_val = torch.cat(val_embeddings, dim=0).numpy()
            y_val = torch.cat(val_labels, dim=0).numpy()
            
            y_pred = knn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted')
            # If you need AUC, ensure binary or one-vs-rest for multi-class.
            print(f"Interim k-NN Eval - Accuracy: {acc*100:.2f}%, F1: {f1:.4f}")
            model.train()
    
    avg_loss = running_loss / (batch_idx + 1)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")


# alt_checkpoints/epoch3_batch1000.pth: 83%