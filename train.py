#!/usr/bin/env python

import os
import argparse
import logging
import numpy as np
import scipy
from time import time
import sys
import pdb
import pickle as pk
# pytorch imports
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# User imports
from src.model import Model
from src.dataset import ASAPDataset, ASAPDataLoader
import src.utils as U

logger = logging.getLogger(__name__)

# parsing arguments

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', required=True, help="The path to the development set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', required=False, help="Promp ID for ASAP dataset. '0' means all prompts.")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp', help="Model type (reg|regp|breg|bregp) (default=regp)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae) (default=mse)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=10, help="Embeddings dimension (default=50)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=2, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=5, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=-1, help="Vocab size (default=4000)")
parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
parser.add_argument("--clip_norm", dest="clip_norm", type=float, metavar='<float>', default=10.0, help="Threshold to clip gradients")
args = parser.parse_args()

out_dir = args.out_dir_path.strip('\r\n')
U.mkdir_p(out_dir + '/preds')

U.set_logger(out_dir)
U.print_args(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# train
train_dataset = ASAPDataset(args.train_path, vocab_file=out_dir + '/vocab.pkl')
vocab = train_dataset.vocab
train_dataset.make_scores_model_friendly()
# test
test_dataset = ASAPDataset(args.test_path, vocab=vocab)
test_dataset.make_scores_model_friendly()
# dev
dev_dataset = ASAPDataset(args.dev_path, vocab=vocab)
dev_dataset.make_scores_model_friendly()

max_seq_length = max(train_dataset.maxlen,
                     test_dataset.maxlen,
                     dev_dataset.maxlen)


def mean0(ls):
    if isinstance(ls[0], list):
        islist = True
        mean = [0.0 for i in range(len(ls[0]))]
    else:
        islist = False
        mean = 0.0
    for i in range(len(ls)):
        if islist:
            for j in range(len(mean)):
                mean[j] += ls[i][j]
        else:
            mean += ls[i]
    if islist:
        for i in range(len(mean)):
            mean[i] /= len(ls)
    else:
        mean /= len(ls)
        mean = [mean]
    return mean


imv = mean0(train_dataset.y)
model = Model(args, vocab, imv)
optimizable_parameters = model.parameters()
loss_fn = F.mse_loss if args.loss == 'mse' else F.l1_loss
optimizer = U.get_optimizer(args, optimizable_parameters)

for epoch in range(args.epochs):
    losses = []
    batch_idx = -1
    for xs, ys, ps, padding_mask, lens in ASAPDataLoader(train_dataset, train_dataset.maxlen, args.batch_size):
        print('Starting batch %d' % batch_idx)
        batch_idx += 1
        # pdb.set_trace()
        youts = model(xs,
                      mask=padding_mask,
                      lens=lens)
        pdb.set_trace()
        loss = 0
        loss = loss_fn(youts, ys)
        losses.append(loss.data[0])
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(optimizable_parameters, args.clip_norm)
        optimizer.step()
        print('\tloss=%f' % (losses[-1]))
        logger.info(
            'Epoch=%d batch=%d loss=%f' % (epoch, batch_idx, losses[-1])
            )
    print('Epoch %d: average loss=%f' % (epoch, sum(losses) / len(losses)))
