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

# model = MedViT_small(num_classes = n_classes).cuda()

# # define loss function and optimizer
# if task == "multi-label, binary-class":
#     criterion = nn.BCEWithLogitsLoss()
# else:
#     criterion = nn.CrossEntropyLoss()

# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# VAL_INTERVAL = 50  # check validation every 50 batches

# for epoch in range(NUM_EPOCHS):
#     train_correct = 0
#     train_total = 0
#     running_loss = 0.0
#     print('Epoch [%d/%d]' % (epoch+1, NUM_EPOCHS))
#     model.train()
#     for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
#         inputs, targets = inputs.cuda(), targets.cuda()
#         optimizer.zero_grad()
#         outputs = model(inputs)
        
#         if task == 'multi-label, binary-class':
#             targets = targets.to(torch.float32)
#             loss = criterion(outputs, targets)
#         else:
#             targets = targets.squeeze().long()
#             loss = criterion(outputs, targets)
        
#         loss.backward()
#         optimizer.step()
        
#         running_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         train_total += targets.size(0)
#         train_correct += (predicted == targets).sum().item()
        
#         # Perform validation check every VAL_INTERVAL batches
#         if (batch_idx + 1) % VAL_INTERVAL == 0:
#             model.eval()
#             val_correct = 0
#             val_total = 0
#             with torch.no_grad():
#                 for val_inputs, val_targets in test_loader:
#                     val_inputs, val_targets = val_inputs.cuda(), val_targets.squeeze().long().cuda()
#                     val_outputs = model(val_inputs)
#                     _, val_predicted = torch.max(val_outputs, 1)
#                     val_total += val_targets.size(0)
#                     val_correct += (val_predicted == val_targets).sum().item()
#             val_acc = 100 * val_correct / val_total
#             print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}], "
#                   f"Loss: {running_loss/(batch_idx+1):.4f}, "
#                   f"Train Acc: {100*train_correct/train_total:.2f}%, "
#                   f"Val Acc: {val_acc:.2f}%")
#             model.train()  # Switch back to training mode

#     # End-of-epoch evaluation on the full test set
#     model.eval()
#     test_correct = 0
#     test_total = 0
#     with torch.no_grad():
#         for inputs, targets in test_loader:
#             inputs, targets = inputs.cuda(), targets.squeeze().long().cuda()
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             test_total += targets.size(0)
#             test_correct += (predicted == targets).sum().item()
#     epoch_val_acc = 100 * test_correct / test_total
#     print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] completed: Train Acc: {100*train_correct/train_total:.2f}%, "
#           f"Val Acc: {epoch_val_acc:.2f}%")





import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pytorch_metric_learning import losses  # use losses from pytorch_metric_learning

# ----- Model Initialization & Pretrained Weights Loading -----
model = MedViT_small(num_classes=n_classes).cuda()

# Load pretrained weights from a specified path.
pretrained_path = '/homes/bhsu/2024_research/med_class/MedViT/weights/MedViT_base_im1k.pth'
state_dict = torch.load(pretrained_path, map_location='cuda')
# Use strict=False because the classifier head weights may be incompatible.
model.load_state_dict(state_dict, strict=False)

# Bypass the classifier head so that forward returns backbone embeddings.
model.proj_head = nn.Identity()

# Optionally, wrap forward to include L2 normalization for metric learning.
def forward_embedding(x):
    emb = model(x)
    return nn.functional.normalize(emb, p=2, dim=1)

# ----- Optimizer & Loss Function Setup -----
# Use a supervised contrastive loss (SupConLoss) which is akin to NTXent/SimCLR loss.
metric_loss_func = losses.SupConLoss(temperature=0.07)

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# Directory to save checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_interval = 100  # Save checkpoint every 10 batches

# ----- Training Loop -----
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        
        # Get embeddings (backbone features)
        embeddings = forward_embedding(inputs)
        # breakpoint()
        # Compute supervised contrastive loss directly on embeddings and labels.
        loss = metric_loss_func(embeddings, targets.squeeze(1))
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Save a checkpoint every 'save_interval' batches.
        if (batch_idx + 1) % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch{epoch+1}_batch{batch_idx+1}.pth')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            print(f'Intermediate Loss: {running_loss / (batch_idx + 1):.4f}')
            
    avg_loss = running_loss / (batch_idx + 1)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.4f}")
    # Optionally, perform evaluation here using the embeddings.

