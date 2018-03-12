'''
    File implements a few neural layers
'''

__author__ = 'haroun habeeb'
__mail__ = 'haroun7@gmail.com'
# general imports
import argparse
import pickle
import os
import sys
import pdb
import numpy as np
import logging
import nltk
import re
# pytorch imports
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli
# Custom imports
from .utils import tensordot

class Conv1DWithMasking(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv1DWithMasking, self).__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
        # Make it easy to initialize/debug.
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, x, mask=None):
        # Shouldn't need because padding is 0
        inp = x.permute([0, 2, 1])
        res = self.conv(inp)
        res = res.permute([0, 2, 1])
        res = res * (mask.unsqueeze(2).expand(*res.size()) if mask is not None else 1)
        return res

class MeanOverTime(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(MeanOverTime, self).__init__()

    def forward(self, x, mask=None, lens=None, dim=1):
        if lens is None:
            return x.mean(dim=dim)
        else:
            s = x.sum(dim=dim)
            return s / lens.unsqueeze(1).expand(*s.size()).float()

    def output_shape(self, input_shape):
        return torch.Size(input_shape[0], input_shape[2])


class Attention(torch.nn.Module):
    def __init__(self, input_size, attention_fn=F.tanh, op='attsum', init_stdev=0.01, *args, **kwargs):
        super(Attention, self).__init__()
        self.input_size = input_size
        if attention_fn is None:
            attention_fn = lambda x: x
        self.attention_fn = attention_fn
        self.op = op
        self.att_V = nn.Parameter(torch.randn(input_size, 1) * init_stdev)
        self.att_W = nn.Parameter(torch.randn(input_size, input_size) * init_stdev)

    def forward(self, x, mask=None, lens=None, dim=1):
        '''
            x: Variable batch_size * max_seq_length * dimensionality
            mask: batch_size * max_seq_length
        '''
        # batch_size * max_seq_length * input_size
        temp = self.attention_fn(tensordot(x, self.att_W))
        # batch_size * max_seq_length * 1
        w = F.softmax(tensordot(temp, self.att_V), dim=2)  #
        # batch_size * max_seq_length * input_size
        w = w.expand(*w.size()[:-1], self.input_size)
        xw = x * w
        s = xw.sum(dim=dim)
        if self.op == 'attsum':
            return s
        elif self.op == 'attmean':
            return s / lens.unsqueeze(1).expand(*s.size()).float()
