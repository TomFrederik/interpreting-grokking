import argparse
import logging
import os
from math import factorial

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

import wandb
from datasets import get_dataset
from model import GrokkingTransformer


def main(
    max_lr,
    min_lr,
    decay_steps,
    warmup_steps,
    weight_decay,
    beta1,
    beta2,
    heads,
    layers,
    width,
    seq_first,
    activation,
    dim_feedforward,
    dropout,
    no_norm,
    tied_embeddings,
    data_name,
    num_elements,
    data_dir,
    force_data,
    no_op_token,
    batch_size,
    steps,
    epochs,
    train_ratio,
    seed,
    verbose,
    log_freq,
    num_workers,
    disable_logging,
    save_every,
    attention_only,
):
    # set wandb logging mode
    if disable_logging:
        os.environ['WANDB_MODE'] = 'offline'
    else:
        os.environ['WANDB_MODE'] = 'online'
    
    # set logging level
    if verbose:
        logging.basicConfig(level=logging.INFO)

    # seeding
    if seed is not None:
        pl.seed_everything(seed)

    # data
    data = get_dataset(descr=data_name, num_elements=num_elements, data_dir=data_dir, force_data=force_data, no_op_token=no_op_token)
    idcs = np.random.permutation(np.arange(len(data)))
    train_idcs = idcs[:int(train_ratio * len(idcs))]
    val_idcs = idcs[int(train_ratio * len(idcs)):]
    if batch_size == -1:
        train_batch_size = len(train_idcs)
        val_batch_size = len(val_idcs)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size
    train_loader = DataLoader(Subset(data, train_idcs), batch_size=train_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(Subset(data, val_idcs), batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # model
    optim_kwargs = {
        'max_lr': max_lr,
        'min_lr': min_lr,
        'warmup_steps': warmup_steps,
        'decay_steps': decay_steps,
        'weight_decay':weight_decay,
        'betas': (beta1, beta2),
    }
    num_tokens = factorial(num_elements) + 2 if data_name.startswith('perm') else num_elements + 2
    max_seq_len = 5
    if no_op_token:
        num_tokens -= 1
        max_seq_len -= 1
        print(f"{num_tokens = }")
    
    model_kwargs = {
        'heads':heads,
        'layers':layers,
        'width':width,
        'max_seq_len':max_seq_len,
        'num_tokens':num_tokens,
        'optim_kwargs':optim_kwargs,
        'dropout':dropout,
        'batch_first':not seq_first,
        'activation':activation,
        'dim_feedforward':dim_feedforward,
        'no_norm':no_norm,
        'tied_embeddings':tied_embeddings,
        'attention_only':attention_only,
    }
    model = GrokkingTransformer(**model_kwargs)

    # wandb config
    config = dict(
        **optim_kwargs, 
        **model_kwargs, 
        batch_size=batch_size, 
        steps=steps, 
        data_name=data_name, 
        num_elements=num_elements,
        seed=seed,
        no_op_token=no_op_token,
    )

    if wandb is None:
        print('wandb')
    
    # callbacks
    callbacks = []
    callbacks.append(
        pl.callbacks.model_checkpoint.ModelCheckpoint(
            verbose=verbose,
            monitor="Validation/Accuracy",
            auto_insert_metric_name=True,
            every_n_epochs=save_every,
            save_top_k=-1,
        )
    )
    callbacks.append(pl.callbacks.progress.TQDMProgressBar(refresh_rate=log_freq))
    callbacks.append(pl.callbacks.lr_monitor.LearningRateMonitor())
    
    # training
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        max_steps=steps,
        max_epochs=epochs,
        log_every_n_steps=log_freq,
        callbacks=callbacks,
        logger=WandbLogger(project="interpreting_grokking", config=config),
    )
    trainer.fit(model, train_loader, val_loader)
    
    callbacks[0].to_yaml()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # optimizer args
    parser.add_argument("--max_lr", type=float, default=0.001)
    parser.add_argument("--min_lr", type=float, default=0.0001)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--decay_steps", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)

    # model args
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--seq_first", action="store_true", help="Whether to have time dim first or batch dim first")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dim_feedforward", type=int, default=None, help='defaults to 4*width')
    parser.add_argument("--no_norm", action='store_true', help='Disables layer norm')
    parser.add_argument("--tied_embeddings", action='store_true', help='Whether to reuse embedding weights as unembedding')
    parser.add_argument("--attention_only", action='store_true', help='Whether to use attention-only trafo')
    

    # data args
    parser.add_argument("--data_name", type=str, default="minus", choices=[
        # Permutation
        "perm_xy", 
        "perm_xyx1",
        "perm_xyx",
        
        # Arithmetic, all operations are mod p
        "plus", # x + y
        "minus", # x - y
        "div", # x / y
        "div_odd", # x / y if y is odd else x - y
        "x2y2", # x^2 + y^2
        "x2xyy2", # x^2 + y^2 + xy
        "x2xyy2x", # x^2 + y^2 + xy + x
        "x3xy", # x^3 + y
        "x3xy2y", # x^3 + xy^2 + y
    ])
    parser.add_argument("--num_elements", type=int, default=97) # choose 5 for permutation data, 97 for arithmetic data
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--force_data", action="store_true", help="Whether to force dataset creation. Do this if you want to make sure that you are loading data with correct settings.")
    parser.add_argument("--no_op_token", action="store_true", help="Whether to add an op token between the number tokens.")
    
    # training args
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10**5)
    parser.add_argument("--epochs", type=int, default=1100)
    parser.add_argument("--train_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_logging", action="store_true")
    parser.add_argument("--save_every", type=int, default=1)
    
    # collect args and run main
    args = parser.parse_args()
    if args.verbose:
        print(f'Arguments = {args}')

    main(**vars(args))
