# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import os
import time
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from args import get_args
from src.transform import MultiVisionDataset
from methods import get_method
from eval.evaluate import get_acc
from eval.dataloader import get_clf

from src.utils import (
    setup_logging,
    init_distributed_device,
    restart_from_checkpoint,
    AverageMeter,
    cosine_scheduler,
    )

import wandb

def random_seed(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def main():
    args = get_args()
    random_seed(args)

    # fully initialize distributed device environment
    device, args = init_distributed_device(args)

    if args.rank == 0: wandb.init(project=args.wandb_project, name = args.env_name, config=args)

    if not os.path.exists(args.dump_path):
        # Create the folder if it doesn't exist
        os.makedirs(args.dump_path)
    setup_logging(os.path.join(args.dump_path,'out.log'), logging.INFO)

    if args.local_rank != 0:
        def log_pass(*args): pass
        logging.info = log_pass

    # build data
    train_dataset = MultiVisionDataset(
        args.data_path,
        args,
        dataset_type=args.dataset,
        download=True,
        return_index=False,
    )
    ds = get_clf(args)

    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    logging.info("Building data done with {} images loaded.".format(len(train_dataset)))

    # build model
    model = get_method(args)

    # synchronize batch norm layers
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # copy model to GPU
    torch.cuda.set_device(device)
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device])

    logging.info(model)
    logging.info("Building model done.")

    # build optimizer
    args.lr = args.lr * args.batch_size * args.world_size / 256

    # ============ init schedulers ... ============
    args.lr_schedule = cosine_scheduler(
        args.lr,
        0.1 * args.lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    args.momentum_schedule = cosine_scheduler(
            args.momentum, 1,
            args.epochs, len(train_loader)
    )

    optimizer = torch.optim.SGD(model.parameters(), 0, momentum=0.9, weight_decay=args.wd)

    logging.info("Building SGD optimizer done.")

    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
    )
    start_epoch = to_restore["epoch"]

    cudnn.benchmark = True

    #scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):

        # train the network for one epoch
        logging.info("============ Starting epoch %i ... ============" % epoch)
        # set sampler
        train_loader.sampler.set_epoch(epoch)

        # train the network
        unwrap_model(model).current_epoch = epoch
        loss = train(train_loader, model, optimizer, epoch, args)

        # save checkpoints
        if args.rank == 0:
            if epoch == 0 or (epoch + 1) % args.eval_every == 0:
                acc_knn, acc = get_acc(model, ds, args)
                logging.info("acc: {}, acc_5: {}, acc_knn: {}".format(acc[1], acc[5], acc_knn))
                wandb.log({"acc": acc[1], "acc_5": acc[5], "acc_knn": acc_knn}, commit=False)

            wandb.log({"learning_rate": optimizer.param_groups[0]['lr'],  
                       "momentum": unwrap_model(model).momentum,
                       "loss": loss, "ep": epoch})

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                save_dict,
                os.path.join(args.dump_path, "checkpoint.pth.tar"),
            )

def train(loader, model, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()

    end = time.time()
    start_idx = 0
    for it, samples in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # update parameters
        iters = len(loader) * epoch + it  # global training iteration
        adjust_parameters(model, optimizer, args, iters)
        
        # ============ backward and optim step ... ============
        optimizer.zero_grad()
        loss = model(samples)
        loss.backward()
        optimizer.step()

        # ============ misc ... ============
        losses.update(loss.item(), samples[0].size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank ==0 and it % 50 == 0:
            logging.info(
                "Epoch: [{0}][{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Lr: {lr:.4f}".format(
                    epoch,
                    it,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    lr=optimizer.param_groups[0]["lr"],
                )
            )
    return losses.avg

def adjust_parameters(model, optimizer, args, iters):
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr_schedule[iters]

    unwrap_model(model).momentum = args.momentum_schedule[iters]

if __name__ == "__main__":
    main()


