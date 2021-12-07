import os
import numpy as np
import torch
import torch.nn.functional as F
import logging

def soft_cross_entropy(input, target, reduction='mean'):
    """
    Cross entropy loss with input logits and soft target
    :param input: Tensor, size: (N, C)
    :param target: Tensor, size: (N, C)
    :param reduction: 'none' or 'mean' or 'sum', default: 'mean'
    :return: loss
    """
    eps = 1.0e-1
    # debug = False
    valid = (target.sum(1) - 1).abs() < eps
    # if debug:
    #     print('valid', valid.sum().item())
    #     print('all', valid.numel())
    #     print('non valid')
    #     print(target[valid == 0])
    if valid.sum().item() == 0:
        return input.new_zeros(())
    if reduction == 'mean':
        return (- F.log_softmax(input[valid], 1) * target[valid]).sum(1).mean(0)
    elif reduction == 'sum':
        return (- F.log_softmax(input[valid], 1) * target[valid]).sum()
    elif reduction == 'none':
        l = input.new_zeros((input.shape[0], ))
        l[valid] = (- F.log_softmax(input[valid], 1) * target[valid]).sum(1)
        return l
    else:
        raise ValueError('Not support reduction type: {}.'.format(reduction))