#!/usr/bin/env python

import argparse
import logging
import numpy as np
import scipy
from time import time
import sys
import src.utils as U
import pickle as pk
from src.asap_evaluator import Evaluator
import src.asap_reader as dataset
from src.models import create_model
from src.optimizers import get_optimizer
import torch.nn.functional as F
from torch.autograd import Variable
from torch import from_numpy
import torch

logger = logging.getLogger(__name__)

###############################################################################################################################
## Parse arguments
#

parser = argparse.ArgumentParser()
parser.add_argument("-tr", "--train", dest="train_path", type=str, metavar='<str>', required=True, help="The path to the training set")
parser.add_argument("-tu", "--tune", dest="dev_path", type=str, metavar='<str>', required=True, help="The path to the development set")
parser.add_argument("-ts", "--test", dest="test_path", type=str, metavar='<str>', required=True, help="The path to the test set")
parser.add_argument("-o", "--out-dir", dest="out_dir_path", type=str, metavar='<str>', required=True, help="The path to the output directory")
parser.add_argument("-p", "--prompt", dest="prompt_id", type=int, metavar='<int>', required=True, help="Promp ID for ASAP dataset. '0' means all prompts.")
parser.add_argument("-t", "--type", dest="model_type", type=str, metavar='<str>', default='regp', help="Model type (reg|regp|breg|bregp) (default=regp)")
parser.add_argument("-u", "--rec-unit", dest="recurrent_unit", type=str, metavar='<str>', default='lstm', help="Recurrent unit type (lstm|gru|simple) (default=lstm)")
parser.add_argument("-a", "--algorithm", dest="algorithm", type=str, metavar='<str>', default='rmsprop', help="Optimization algorithm (rmsprop|sgd|adagrad|adadelta|adam|adamax) (default=rmsprop)")
parser.add_argument("-l", "--loss", dest="loss", type=str, metavar='<str>', default='mse', help="Loss function (mse|mae) (default=mse)")
parser.add_argument("-e", "--embdim", dest="emb_dim", type=int, metavar='<int>', default=50, help="Embeddings dimension (default=50)")
parser.add_argument("-c", "--cnndim", dest="cnn_dim", type=int, metavar='<int>', default=0, help="CNN output dimension. '0' means no CNN layer (default=0)")
parser.add_argument("-w", "--cnnwin", dest="cnn_window_size", type=int, metavar='<int>', default=3, help="CNN window size. (default=3)")
parser.add_argument("-r", "--rnndim", dest="rnn_dim", type=int, metavar='<int>', default=300, help="RNN dimension. '0' means no RNN layer (default=300)")
parser.add_argument("-b", "--batch-size", dest="batch_size", type=int, metavar='<int>', default=32, help="Batch size (default=32)")
parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
parser.add_argument("--aggregation", dest="aggregation", type=str, metavar='<str>', default='mot', help="The aggregation method for regp and bregp types (mot|attsum|attmean) (default=mot)")
parser.add_argument("--dropout", dest="dropout_prob", type=float, metavar='<float>', default=0.5, help="The dropout probability. To disable, give a negative number (default=0.5)")
parser.add_argument("--vocab-path", dest="vocab_path", type=str, metavar='<str>', help="(Optional) The path to the existing vocab file (*.pkl)")
parser.add_argument("--skip-init-bias", dest="skip_init_bias", action='store_true', help="Skip initialization of the last layer bias")
parser.add_argument("--emb", dest="emb_path", type=str, metavar='<str>', help="The path to the word embeddings file (Word2Vec format)")
parser.add_argument("--epochs", dest="epochs", type=int, metavar='<int>', default=50, help="Number of epochs (default=50)")
parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
parser.add_argument("--seed", dest="seed", type=int, metavar='<int>', default=1234, help="Random seed (default=1234)")
args = parser.parse_args()

out_dir = args.out_dir_path

U.mkdir_p(out_dir + '/preds')
U.set_logger(out_dir)
U.print_args(args)

assert args.model_type in {'reg', 'regp', 'breg', 'bregp'}
assert args.algorithm in {'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adam', 'adamax'}
assert args.loss in {'mse', 'mae'}
assert args.recurrent_unit in {'lstm', 'gru', 'simple'}
assert args.aggregation in {'mot', 'attsum', 'attmean'}

if args.seed > 0:
    np.random.seed(args.seed)

###############################################################################################################################
## Prepare data
#

def pad_sequences(vectorized_seqs, value=0.0, maxlen=None):
    if maxlen is None:
        maxlen = max(len(i) for i in vectorized_seqs)
    # get the length of each seq in your batch
    seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))

    # dump padding everywhere, and place seqs on the left.
    # NOTE: you only need a tensor as big as your longest sequence
    seq_tensor = Variable(torch.zeros((len(vectorized_seqs), maxlen)), requires_grad=False).long()
    for idx, (seq, seqlen) in enumerate(zip(vectorized_seqs, seq_lengths)):
        maxi = min(seqlen,maxlen)
        seq_tensor[idx, :maxi] = torch.LongTensor(seq[:maxi])

    # SORT YOUR TENSORS BY LENGTH!
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    seq_tensor = seq_tensor.transpose(0,1)

    return seq_tensor, seq_lengths

# data_x is a list of lists
(train_x, train_y, train_pmt), (dev_x, dev_y, dev_pmt), (test_x, test_y, test_pmt), vocab, vocab_size, overal_maxlen, num_outputs = dataset.get_data(
    (args.train_path, args.dev_path, args.test_path), args.prompt_id, args.vocab_size, args.maxlen, tokenize_text=True, to_lower=True, sort_by_len=False, vocab_path=args.vocab_path)

# Dump vocab
with open(out_dir + '/vocab.pkl', 'wb') as vocab_file:
    pk.dump(vocab, vocab_file)

# Pad sequences for mini-batch processing
if args.model_type in {'breg', 'bregp'}:
    assert args.rnn_dim > 0
    assert args.recurrent_unit == 'lstm'
    train_x_tuple = pad_sequences(train_x, maxlen=overal_maxlen)
    dev_x_tuple = pad_sequences(dev_x, maxlen=overal_maxlen)
    test_x_tuple = pad_sequences(test_x, maxlen=overal_maxlen)
else:
    train_x_tuple = pad_sequences(train_x, value=0.)
    dev_x_tuple = pad_sequences(dev_x, value=0.)
    test_x_tuple = pad_sequences(test_x, value=0.)

###############################################################################################################################
## Some statistics
#

train_y = np.array(train_y, dtype='float32')
dev_y = np.array(dev_y, dtype='float32')
test_y = np.array(test_y, dtype='float32')

train_pmt = np.array(train_pmt, dtype='int32')
dev_pmt = np.array(dev_pmt, dtype='int32')
test_pmt = np.array(test_pmt, dtype='int32')

bincounts, mfs_list = U.bincounts(train_y)
with open('%s/bincounts.txt' % out_dir, 'w') as output_file:
    for bincount in bincounts:
        output_file.write(str(bincount) + '\n')

train_mean = train_y.mean(axis=0)
train_std = train_y.std(axis=0)
dev_mean = dev_y.mean(axis=0)
dev_std = dev_y.std(axis=0)
test_mean = test_y.mean(axis=0)
test_std = test_y.std(axis=0)

logger.info('Statistics:')

# logger.info('  train_x shape: ' + str(np.array(train_x).shape))
# logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
# logger.info('  test_x shape:  ' + str(np.array(test_x).shape))

logger.info('  train_y shape: ' + str(train_y.shape))
logger.info('  dev_y shape:   ' + str(dev_y.shape))
logger.info('  test_y shape:  ' + str(test_y.shape))

logger.info('  train_y mean: %s, stdev: %s, MFC: %s' % (str(train_mean), str(train_std), str(mfs_list)))

# We need the dev and test sets in the original scale for evaluation
dev_y_org = dev_y.astype(dataset.get_ref_dtype())
test_y_org = test_y.astype(dataset.get_ref_dtype())

# Convert scores to boundary of [0 1] for training and evaluation (loss calculation)
train_y = dataset.get_model_friendly_scores(train_y, train_pmt)
dev_y = dataset.get_model_friendly_scores(dev_y, dev_pmt)
test_y = dataset.get_model_friendly_scores(test_y, test_pmt)

###############################################################################################################################
## Building model
#

if args.loss == 'mse':
    loss = F.mse_loss
    metric = F.l1_loss
else:
    loss = F.l1_loss
    metric = F.mse_loss
labels = [i / 10.0 for i in range(11)]
model = create_model(args, len(labels), np.array(train_y).mean(), overal_maxlen, vocab)
optimizer = get_optimizer(args.algorithm, model)

###############################################################################################################################
## Plotting model - taking out
#

###############################################################################################################################
## Save model architecture - taking out
#

###############################################################################################################################
## Training
#

logger.info('--------------------------------------------------------------------------------------------------------------------------')
logger.info('Initial Evaluation:')

total_train_time = 0
total_eval_time = 0

mapping = dict((l, i) for i, l in enumerate(labels))
def one_hot(np_y, mapping):
    zeros = torch.zeros((np_y.shape[0], len(mapping)))
    for i, val in enumerate(np_y):
        zeros[i, mapping[val]] = 1
    return Variable(zeros).float()

train_y = one_hot(train_y, mapping)
dev_y = one_hot(dev_y, mapping)
test_y = one_hot(test_y, mapping)

train_pmt = from_numpy(train_pmt)
dev_pmt = from_numpy(dev_pmt)
test_pmt = from_numpy(test_pmt)


def train(model, optimizer):
    model.train()
    data, target = train_x_tuple, train_y
    model.zero_grad()
    output = model(*data)
    optimizer.zero_grad()
    loss_value = loss(output, target)
    loss_value.backward()
    optimizer.step()
    return loss_value


for ii in range(args.epochs):
    # TODO: Gradient clipping has to be done here instead of the optimizer

    # Training
    t0 = time()
    calc_loss = train(model, optimizer)
    tr_time = time() - t0
    total_train_time += tr_time

    # Evaluate
    t0 = time()
    #evl.evaluate(model, ii)
    evl_time = time() - t0
    total_eval_time += evl_time

    # Print information
    logger.info('Epoch %d, train: %is, evaluation: %is' % (ii, tr_time, evl_time))
    logger.info('[Train] loss: %.4f' % (calc_loss,))
    #evl.print_info()

###############################################################################################################################
## Summary of the results
#

logger.info('Training:   %i seconds in total' % total_train_time)
logger.info('Evaluation: %i seconds in total' % total_eval_time)

# evl.print_final_info()