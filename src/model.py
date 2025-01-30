# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch.nn as nn
import src.resnet as resnet

def get_projector(out_size, args):
    """ creates projector g() from config """
    x = []
    in_size = out_size
    for _ in range(args.mlp_layers - 1):
        x.append(nn.Linear(in_size, args.mlp_dim))
        x.append(nn.BatchNorm1d(args.mlp_dim))
        x.append(nn.ReLU(inplace=True))
        in_size = args.mlp_dim
        
    x.append(nn.Linear(in_size, args.emb))
    x.append(nn.BatchNorm1d(args.emb, affine=False))

    projector = nn.Sequential(*x)

    return projector


def get_encoder(args):
    """ creates encoder E() by name and modifies it for dataset """

    encoder, out_size = resnet.__dict__[args.arch](
            zero_init_residual=(args.arch != 'resnet18'))
    
    if 'cifar' in args.dataset:
        encoder.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        encoder.maxpool = nn.Identity()
                
    return encoder, out_size
