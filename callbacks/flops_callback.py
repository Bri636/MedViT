""" Custom flops callback for model throuput """

from __future__ import annotations

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.utilities import measure_flops


class FlopsLoggerCallback(Callback):
    def __init__(self):
        super().__init__()

    def setup(self, trainer, pl_module, stage):
        # Ensure the logger is a WandbLogger
        if not isinstance(trainer.logger, WandbLogger):
            raise TypeError("Trainer logger must be a WandbLogger instance.")
        # Create a meta-device model for FLOPs calculation
        with torch.device('meta'):
            meta_model = pl_module.__class__(*pl_module.args, **pl_module.kwargs)
        # Generate a sample input tensor with the same shape as your data
        input_sample = torch.randn(1, 3, 224, 224, device='meta')  # Adjust dimensions as needed
        # Calculate FLOPs using the measure_flops utility
        flops = measure_flops(meta_model, input_sample)
        # Log FLOPs to Weights & Biases
        trainer.logger.experiment.config.update({"FLOPs": flops})