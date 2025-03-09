""" Main Training Script """
from __future__ import annotations

import os
import torch
import pprint as pp
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, ThroughputMonitor
from pytorch_lightning.loggers import WandbLogger
from pydantic import Field
from pathlib import Path
from argparse import ArgumentParser, Namespace
import time
import itertools
from torch import nn
# me packages 
# callbacks
from callbacks.knn_callback import KNNCallBackConfig, KNN_Evaluation_Callback
from callbacks.flops_callback import FlopsLoggerCallback
from medmnist_datasets.medmnist_dataset import make_dataloaders
from utils import BaseConfig
from VitLightning import LitMedViT

CURRENT_DIR = Path(__file__).resolve().parent
IM1K_WEIGHTS = CURRENT_DIR / Path('weights/MedViT_base_im1k.pth')
# some training defaults
class TrainConfig(BaseConfig): 
    """ Training config """
    model_checkpoint_path: str = IM1K_WEIGHTS
    save_checkpoint_dir: str = '/homes/bhsu/2024_research/MedViT/models_all_checkpoints/test_lightning_checkpoints/'
    wandb_log_proj: str = "MedViT_KNN_Eval"
    wandb_log_offline: bool = True
    data_flag: str = 'octmnist'
    strategy: str = 'ddp'
    num_epochs: int = 100
    lr: float = 0.005
    batch_size: int = 64
    knn_config: KNNCallBackConfig = Field(default_factory=KNNCallBackConfig)
    every_n_train_steps: int = 100 # save checkpoint every n batches
    log_every_n_steps: int = 100 # log every 100 steps 
    
def parse_arguments() -> Namespace: 
    argparser = ArgumentParser()
    argparser.add_argument('--model_checkpoint_path', 
                           type=str, 
                           default='/homes/bhsu/2024_research/MedViT/models_all_checkpoints/test_lightning_checkpoints/epoch=1-step=500.ckpt')
    argparser.add_argument('--train_config_path', 
                           type=str, 
                           default='/homes/bhsu/2024_research/MedViT/configs/lambda_train_config.yaml')
    argparser.add_argument('--save_checkpoint_dir', 
                           type=str, 
                           default='/homes/bhsu/2024_research/MedViT/models_all_checkpoints/test_lightning_checkpoints/')
    argparser.add_argument('--train_from_scratch', 
                           action='store_true')
    argparser.add_argument('--debug', 
                           action='store_true')
    argparser.add_argument('--log', 
                           action='store_true')
    return argparser.parse_args()

    
def main(): 
    args = parse_arguments()
    print(f'Reading in config from here: {args.train_config_path}')
    train_config = TrainConfig.from_yaml(args.train_config_path)
    train_config.model_checkpoint_path = args.model_checkpoint_path # update config
    train_config.save_checkpoint_dir = args.save_checkpoint_dir
    
    wandb_logger = WandbLogger(project=train_config.wandb_log_proj, offline=not args.log)
    datasets = make_dataloaders(train_config.data_flag, train_config.batch_size)
    
    train_dataloader = datasets['train']
    val_dataloader = datasets['validation']
    n_classes = datasets['n_classes']
    train_dataloader = itertools.islice(train_dataloader, 2) if args.debug else train_dataloader # for debugging
    
    model = LitMedViT(n_classes=n_classes, 
                      lr=train_config.lr, 
                      pretrained_path=train_config.model_checkpoint_path)

    knn_callback = KNN_Evaluation_Callback(train_dataloader=train_dataloader, 
                                           val_dataloader=val_dataloader, 
                                           k=train_config.knn_config.k, 
                                           log_every_n_steps=train_config.knn_config.log_every_n_steps, 
                                           max_train_batches=train_config.knn_config.max_train_batches)
    # (Optional) Checkpoint callback to save model every 100 training steps.
    # NOTE: IMPORTANT - dirpath should match model_checkpoint_path
    if not args.train_from_scratch: 
        assert Path('/'.join(train_config.model_checkpoint_path.split('/')[:-1]))==Path(train_config.save_checkpoint_dir), f"""
    STOP: loaded checkpoint_weight_path: {train_config.model_checkpoint_path} should probably be saved 
    in the same directory: {train_config.save_checkpoint_dir}  
    """
    print('==================================================================')
    print(f'Checkpoint loaded from here: {train_config.model_checkpoint_path}')
    print('==================================================================')
    print(f'Will save to here: {train_config.save_checkpoint_dir}')
    print('==================================================================')
    print(f'Reading in Config from here: {args.train_config_path}')
    print('==================================================================')
    print(f'{pp.pformat(train_config.model_dump())}')
    print('==================================================================')
    time.sleep(1)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=train_config.save_checkpoint_dir, # 
        filename="{epoch}-{step}",
        every_n_train_steps=train_config.every_n_train_steps # every 100 batches
    )
    
    trainer = pl.Trainer(
        strategy=train_config.strategy,
        max_epochs=train_config.num_epochs,
        devices='auto',
        logger=wandb_logger,
        callbacks=[knn_callback, checkpoint_callback],
        log_every_n_steps=train_config.log_every_n_steps,
        num_sanity_val_steps=0 # avoid validation hangup 
    )
    if args.train_from_scratch: 
        print(f'WARNING: YOU ARE TRAINING FROM SCRATCH FROM HERE: {IM1K_WEIGHTS}\nAND SAVING HERE: {train_config.save_checkpoint_dir}')
        time.sleep(5)
        trainer.fit(model, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader) # no ckpt needed; from scratch
    else: 
        trainer.fit(model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader, 
                ckpt_path=train_config.model_checkpoint_path) # start from this checkpoint
    
if __name__ == "__main__": 
    main()
    