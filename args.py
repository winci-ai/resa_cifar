# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import argparse
from torchvision import models

def get_args():
    parser = argparse.ArgumentParser(description="Implementation of ReSA")

    parser.add_argument("--wandb_project", type=str, default="resa_cifar",
                    help="name of the run for wandb project" )

    parser.add_argument("--env_name", type=str, default="resa_cifar",
                    help="name of the run for wandb env" )

    parser.add_argument("--dump_path", type=str, default=".",
                        help="experiment dump path for checkpoints and log")

    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

    #########################
    #### data parameters ####
    #########################
    parser.add_argument("--dataset", type=str, default="cifar10")

    parser.add_argument("--data_path", type=str, default="./data",
                    help="path to dataset repository")

    parser.add_argument("--crops_nmb", type=int, default=[1], nargs="+",
                    help="list of number of crops")

    parser.add_argument("--crops_size", type=int, default=[32], nargs="+",
                    help="crops resolutions")

    parser.add_argument("--solarization_prob", type=float, default=[0.2], nargs="+",
                    help="solarization prob")

    parser.add_argument("--size_dataset", type=int, default=-1, 
                    help="size of dataset")

    parser.add_argument("--workers", default=4, type=int,
                    help="number of data loading workers")
    
    #########################
    ## resa specific params #
    #########################

    parser.add_argument("--temperature", default=0.4, type=float,
                    help="temperature parameter in training loss")

    parser.add_argument("--momentum", type=float, default=0.996, 
                    help="momentum")

    #########################
    #### optim parameters ###
    #########################
    parser.add_argument("--epochs", default=1000, type=int,
                    help="number of total epochs to run")

    parser.add_argument("--batch_size", default=256, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")

    parser.add_argument('--lr', default=0.3, type=float, 
                    help='initial (base) learning rate for train')

    parser.add_argument('--wd', default=1e-4, type=float, 
                    help='weight decay (default: 1e-4)')

    parser.add_argument("--warmup_epochs", default=2, type=int, 
                    help="number of warmup epochs")

    #########################
    #### dist parameters ###
    #########################
    parser.add_argument("--world_size", default=1, type=int, 
                    help="""number of processes: it is set automatically and
                            should not be passed as argument""")

    parser.add_argument("--rank", default=0, type=int, 
                    help="rank of this process: it is set automatically and should not be passed as argument")

    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
                    
    parser.add_argument("--no-set-device-rank", default=False, action="store_true",
                    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).")

    parser.add_argument("--dist-url", default="env://", type=str,
                    help="url used to set up distributed training")

    #########################
    #### other parameters ###
    #########################
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (e.g. resnet18, resnet50, resnet200, resnet50x2)')

    parser.add_argument("--mlp_layers", type=int, default=3, 
                    help="number of FC layers in projection")

    parser.add_argument("--mlp_dim", type=int, default=2048, 
                    help="size of FC layers in projection/predictor")

    parser.add_argument("--emb", type=int, default=512, 
                    help="embedding size")

    parser.add_argument(
        "--eval_every", type=int, default=5, help="how often to evaluate"
    )

    return parser.parse_args()