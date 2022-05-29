import platform
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import gc
import os
import wandb
import json
import glob
from scipy import sparse
from pathlib import Path

import torch
import transformers
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader

import warnings
from Trainer import Trainer
from config import Config
from data import MarkdownDataset
import utils as ut
from model import BERTModel
warnings.simplefilter('ignore')



def yield_optimizer(model):
    """
    Returns optimizer for specific parameters
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.003,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return transformers.AdamW(optimizer_parameters, lr=Config.LR)

if __name__ == '__main__':
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
        DEVICE = torch.device('cuda:0')
    else:
        print("\n[INFO] GPU not found. Using CPU: {}\n".format(platform.processor()))
        DEVICE = torch.device('cpu')

    Config.DEVICE = torch.device('cpu')
    triplets,val_triplets,cell_id_to_source = ut.get_data()
    train_set = MarkdownDataset(triplets,cell_id_to_source)
    valid_set = MarkdownDataset(val_triplets,cell_id_to_source)

    train_loader = DataLoader(
        train_set,
        batch_size = Config.TRAIN_BS,
        shuffle = True,
        num_workers = 8
    )

    valid_loader = DataLoader(
        valid_set,
        batch_size = Config.VALID_BS,
        shuffle = False,
        num_workers = 8
    )

    model = BERTModel().to(DEVICE)
    nb_train_steps = int(len(train_set) / Config.TRAIN_BS * Config.NB_EPOCHS)
    optimizer = yield_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=Config.T_0, 
        eta_min=Config.η_min
    )
    train_loss_fn, valid_loss_fn = nn.BCELoss(), nn.BCELoss()
    

    
    trainer = Trainer(
        config = Config,
        dataloaders = (train_loader, valid_loader),
        loss_fns = (train_loss_fn, valid_loss_fn),
        optimizer = optimizer,
        model = model,
        scheduler = scheduler,
        device="cpu"
    )

    best_pred = trainer.fit(
        epochs = Config.NB_EPOCHS,
        custom_name = f"ai4code_distillbert.bin"
    )