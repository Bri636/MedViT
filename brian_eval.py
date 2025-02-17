
# import os
# import sys
# import numpy as np
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data

# import torchvision.utils
# from torchvision import models
# import torchvision.datasets as dsets
# import torchvision.transforms as transforms
# from torchsummary import summary

# from tqdm import tqdm
# import medmnist
# from medmnist import INFO, Evaluator

# import torchattacks
# from torchattacks import PGD, FGSM


# data_flag = 'octmnist'
# # [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
# # pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist, organamnist, organcmnist, organsmnist]
# download = True

# NUM_EPOCHS = 10
# BATCH_SIZE = 10
# lr = 0.005

# info = INFO[data_flag]
# task = info['task']
# n_channels = info['n_channels']
# n_classes = len(info['label'])

# DataClass = getattr(medmnist, info['python_class'])

# print("number of channels : ", n_channels)
# print("number of classes : ", n_classes)

# from torchvision.transforms.transforms import Resize
# # preprocessing
# train_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.Lambda(lambda image: image.convert('RGB')),
#     torchvision.transforms.AugMix(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(224),
#     transforms.Lambda(lambda image: image.convert('RGB')),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[.5], std=[.5])
# ])

# # load the data
# train_dataset = DataClass(split='train', transform=train_transform, download=download)
# test_dataset = DataClass(split='test', transform=test_transform, download=download)

# # pil_dataset = DataClass(split='train', download=download)

# # encapsulate data into dataloader form
# train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
# # breakpoint()

# print(train_dataset)
# print("===================")
# print(test_dataset)
     
# from MedViT import MedViT_small, MedViT_base, MedViT_large

# model = MedViT_small(num_classes = n_classes).cuda()
# #model = MedViT_base(num_classes = n_classes).cuda()
# #model = MedViT_large(num_classes = n_classes).cuda()

# split = 'test'

# model.eval()
# # Initialize tensors for predictions and true labels
# y_true = torch.tensor([], dtype=torch.long)
# y_score = torch.tensor([])

# # Since split is 'test', we use the test_loader (which covers the whole test dataset)
# data_loader = test_loader
# # data_loader = train_loader # NOTE: change

# # NOTE: MINE
# from torch.utils.data import Subset
# subset_indices = torch.arange(10000)  # First 1000 samples
# # Create a subset
# subset_dataset = Subset(train_dataset, subset_indices)
# data_loader = data.DataLoader(dataset=subset_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

# with torch.no_grad():
#     for inputs, targets in tqdm(data_loader, desc="Evaluating on test dataset"):
#         # Move inputs (and targets if needed) to GPU
#         inputs = inputs.cuda()
#         targets = targets.cuda()
        
#         # Get model outputs and convert to probabilities
#         outputs = model(inputs)
#         outputs = outputs.softmax(dim=-1)
        
#         # Collect predictions and true labels
#         y_score = torch.cat((y_score, outputs.cpu()), 0)
#         y_true = torch.cat((y_true, targets.cpu()), 0)

# # Convert tensors to numpy arrays for evaluation
# y_score = y_score.detach().numpy()
# y_true = y_true.detach().numpy()

# # Initialize evaluator (adjust parameters if needed)
# evaluator = Evaluator(data_flag, 'test', size=224)

# # Evaluate using the predictions. Some Evaluator implementations only need y_score.
# metrics = evaluator.evaluate(y_score)

# print('Test  AUC: %.3f  Acc: %.3f' % (metrics[0], metrics[1]))






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


data_flag = 'octmnist'
# [tissuemnist, pathmnist, chestmnist, dermamnist, octmnist,
#  pnemoniamnist, retinamnist, breastmnist, bloodmnist, tissuemnist,
#  organamnist, organcmnist, organsmnist]
download = True

NUM_EPOCHS = 10
BATCH_SIZE = 10
lr = 0.005

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

print("number of channels : ", n_channels)
print("number of classes : ", n_classes)

from torchvision.transforms.transforms import Resize
# Preprocessing
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

# Load the data
train_dataset = DataClass(split='train', transform=train_transform, download=download)
test_dataset = DataClass(split='test', transform=test_transform, download=download)

# Encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)

from MedViT import MedViT_small, MedViT_base, MedViT_large

model = MedViT_small(num_classes=n_classes).cuda()
# model = MedViT_base(num_classes=n_classes).cuda()
# model = MedViT_large(num_classes=n_classes).cuda()

# Change the split to 'train' for evaluating on the training set.
split = 'train'

model.eval()
# Initialize tensors for predictions and true labels
y_true = torch.tensor([], dtype=torch.long)
y_score = torch.tensor([])

# Use the train_loader_at_eval since we are evaluating on the train set.
data_loader = train_loader_at_eval

with torch.no_grad():
    for inputs, targets in tqdm(data_loader, desc="Evaluating on train dataset"):
        # Move inputs (and targets if needed) to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()
        
        # Get model outputs and convert to probabilities
        outputs = model(inputs)
        outputs = outputs.softmax(dim=-1)
        
        # Collect predictions and true labels
        y_score = torch.cat((y_score, outputs.cpu()), 0)
        y_true = torch.cat((y_true, targets.cpu()), 0)

# Convert tensors to numpy arrays for evaluation
y_score = y_score.detach().numpy()
y_true = y_true.detach().numpy()

# Initialize evaluator for the train split
evaluator = Evaluator(data_flag, 'train', size=224)

# Evaluate using the predictions.
metrics = evaluator.evaluate(y_score)

print('Train  AUC: %.3f  Acc: %.3f' % (metrics[0], metrics[1]))


