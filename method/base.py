# Copyright (c) Winci.
# Licensed under the Apache License, Version 2.0 (the "License");

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from src.model import get_encoder, get_projector

class BaseMethod(nn.Module):
    """
        Base class for self-supervised loss implementation.
        It includes encoder and projector for training function.
    """
    def __init__(self, args):
        super().__init__()
        self.encoder, self.out_size = get_encoder(args)
        self.projector = get_projector(self.out_size, args)
        self.momentum = args.momentum
        self.device = args.device

    def ForwardWrapper(self, samples, encoder, projector):

        # do not concate different views if BN is in the model 
        # As it will disrupt the zero-mean, unit-variance distribution
        h = [encoder(x) for x in samples]
        emb = [projector(x) for x in h]
        
        emb = [FullGather.apply(F.normalize(x)) for x in emb]

        with torch.no_grad():
            h = FullGather.apply(F.normalize(h[0]))

        return h, emb

    @torch.no_grad()
    def sinkhorn(self, scores, eps=0.05, niters=3):  # scores must be a square matrix here
        Q = torch.exp(scores / eps).T
        Q /= Q.sum()
        m , _ = Q.shape
        c = torch.ones(m, device=self.device) / m
        for _ in range(niters):
            u = Q.sum(dim=1)
            Q *= (c / u).unsqueeze(1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(0)
        return (Q / Q.sum(dim=0, keepdim=True)).T

    def cross_entropy(self, s, q):
        loss = torch.sum(q * F.log_softmax(s, dim=1), dim=-1).mean() + \
               torch.sum(q.T * F.log_softmax(s.T, dim=1), dim=-1).mean()
        return - loss / 2

    @torch.no_grad()
    def update_momentum_params(self, m):
        """
        Update of the momentum encoder and projector
        """
        for param_q, param_k in zip(self.encoder.parameters(), 
                                    self.momentum_encoder.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        for param_q, param_k in zip(self.projector.parameters(),
                                    self.momentum_projector.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1. - m)
        
    def forward(self, samples):
        raise NotImplementedError

class FullGather(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """
    @staticmethod
    def forward(ctx, input):
        ctx.batch_size = input.shape[0]
        gather_list = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, input)
        return torch.cat(gather_list, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        batch_size = ctx.batch_size
        rank = dist.get_rank()
        grad_input = grad_output[rank * batch_size : (rank + 1) * batch_size]
        return grad_input
