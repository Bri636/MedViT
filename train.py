""" My training implementation for NTXent-like loss with KMeans Clustering """

from __future__ import annotations

import os
import torch
from torch import nn, optim
from tqdm import tqdm
from pytorch_metric_learning import losses  # using SupConLoss for contrastive training
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import itertools
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
from torchsummary import summary
from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import dataclass
from typing import Literal, Callable
import lightning as L
from pydantic import Field
from pathlib import Path
from pytorch_lightning.callbacks import Callback, ModelCheckpoint

# custom packages
import medmnist
from medmnist import INFO, Evaluator
from MedViT import MedViT_small
from utils import BaseConfig
from lightning_model import ViTLightning, ModelConfig
from callbacks import KNN_Evaluation_Callback

# fixing some defaults
CURRENT_DIR = Path(__file__).resolve().parent
# CHECKPOINT_PATH = CURRENT_DIR / Path('./../icor-codon-optimization/training/rjfinalfaa.faa')
# CSV_FILE_PATH = CURRENT_DIR / Path('fasta_converted_sequences.csv')

_OPTIMIZERS = Literal['adamw']
_SCHEDULERS = Literal['linear']
_DATA_FLAGS = Literal[
    'tissuemnist', 'pathmnist', 'chestmnist', 'dermamnist', 'octmnist',
    'pnemoniamnist', 'retinamnist', 'breastmnist', 'bloodmnist',
    'organamnist', 'organcmnist', 'organsmnist'
]

class LightningConfig(BaseConfig): 
    default_root_dir: str = CURRENT_DIR + 'checkpoints/'
    
class TrainConfig(BaseConfig): 
    data_flag: _DATA_FLAGS = 'octmnist'
    batch_size: int = 64
    """ Training batch size """
    num_epochs: int = 10
    lr: float = 1e-4
    optimizer: str = 'adamw'
    lr_scheduler: str = 'linear'
    model_config: ModelConfig = Field(default_factory=ModelConfig)
    trainer_config: ...
    save_every_n: int = 100

def parse_arguments(): 
    argparser = ArgumentParser()
    argparser.add_argument('--model_checkpoint_path', 
                           type=str, 
                           default='/homes/bhsu/2024_research/med_class/MedViT/weights/MedViT_base_im1k.pth')
    argparser.add_argument('--train_config_path', 
                           type=str, 
                           default='./')
    argparser.add_argument('--debug', 
                           action='store_true')
    return argparser.parse_args()

# transformations to apply during training
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

def make_callbacks(train_config: TrainConfig, train_dataloader, val_dataloader) -> list[Callable]: 

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"20M-{train_config.num_epochs}-Epochs-{train_config.batch_size}-BZ", # param-epochs-batch_size
        filename="{epoch}-{step}",
        every_n_train_steps=train_config.save_every_n
    )
    knn_callback = KNN_Evaluation_Callback(train_dataloader, val_dataloader, k=5)
    
    return [checkpoint_callback, knn_callback]
    

def main(): 
    """ Setup datasets and model for training """
    
    args = parse_arguments()
    train_config = TrainConfig.from_yaml(args.train_config_path)
    
    info = INFO[train_config.data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    print(f"Getting Dataset: {train_config.data_flag} For Task: {task}")
    print("Number of channels : ", n_channels)
    print("Number of classes : ", n_classes)
    # creating datasets
    train_dataset = DataClass(split='train', transform=train_transform, download=True)
    test_dataset = DataClass(split='test', transform=test_transform, download=True)   
    # TODO: eventually figure out how to split train into validation 
    # https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987 
    train_loader = DataLoader(dataset=train_dataset, batch_size=train_config.batch_size, shuffle=True)
    train_loader_at_eval = DataLoader(dataset=train_dataset, batch_size= 2 * train_config.batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size= 2 * train_config.batch_size, shuffle=False)  
    
    train_loader = itertools.islice(train_loader, 2) if args.debug else train_loader # for debugging
    
    print(train_dataset)
    print("===================")
    print(test_dataset)
    
    # model setup  
    model = ViTLightning(train_config.model_config)
    print(f"Starting Training With These Configurations: {train_config.model_dump()}")
    
    callbacks = make_callbacks(train_config, train_loader_at_eval, test_loader)
    # TODO: make your callbacks for validation and configs 
    trainer = L.Trainer(callbacks=callbacks)
    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders=train_loader_at_eval, 
                ckpt_path=args.model_checkpoint_path)

if __name__=="__main__": 
    main()