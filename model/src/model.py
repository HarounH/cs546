'''
    Implements the model, given a bunch of options and parameters
    ifmain tests the model too.
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# User imports
from .custom_layers import Conv1DWithMasking, MeanOverTime, Attention
from .embedding_reader import EmbeddingReader
from .dataset import pos_dim

logger = logging.getLogger(__name__)
if sys.platform in ['win32']:
    print('Detected windows OS')
    tsv_encoding = 'latin-1'
    lineneding = '\n'
else:
    tsv_encoding = None
    lineneding = '\n'


class Model(torch.nn.Module):
    def __init__(self, args, vocab, initial_mean_value):
        '''
            args: ArgumentParser.parse_args output thing
        '''
        super(Model, self).__init__()
        self.args = args
        self.vocab = vocab
        if args.recurrent_unit == 'lstm':
            chosen_RNN_class = nn.LSTM
        elif args.recurrent_unit == 'gru':
            chosen_RNN_class = nn.GRU
            raise NotImplementedError('Use LSTM')
        else:
            chosen_RNN_class = nn.RNN
            raise NotImplementedError('Use LSTM')

        pooling = False
        bidirectional_rnn = False
        if args.model_type == 'cls':
            raise NotImplementedError
        elif args.model_type == 'reg':
            pass
        elif args.model_type == 'regp':
            pooling = True
        elif args.model_type == 'breg':
            bidirectional_rnn = True
        elif args.model_type == 'bregp':
            pooling = True
            bidirectional_rnn = True
        else:
            pdb.set_trace()
            raise NotImplementedError

        dropout_W = 0.5         # default=0.5
        dropout_U = 0.1         # default=0.1
        num_outputs = len(initial_mean_value)
        imv = torch.FloatTensor(initial_mean_value)
        init_bias_value = torch.log(imv) - torch.log(1 - imv)
        layers = []
        current_dim = args.vocab_size if args.vocab_size > 0 else len(vocab)

        emb_dim = args.emb_dim
        layers.append(nn.Embedding(current_dim, args.emb_dim))
        self.embedding_layer = layers[-1]
        current_dim = args.emb_dim
        if self.args.pos:
            current_dim += pos_dim()
        if args.cnn_dim > 0:
            if args.cnn_window_size % 2 == 0:
                logger.error('CNN Window size must be odd for\
                             this to work. To maintain the same size as input\
                             sequence, certain restrictions must be made.'
                             )
                raise RuntimeError('cnn size not odd')
            layers.append(
                Conv1DWithMasking(current_dim,
                                  args.cnn_dim,
                                  args.cnn_window_size,
                                  padding=(args.cnn_window_size - 1)//2)
                )
            self.cnn_layer = layers[-1]
            current_dim = args.cnn_dim
        if args.rnn_dim > 0:
            layers.append(
                chosen_RNN_class(
                    input_size=current_dim,
                    hidden_size=args.rnn_dim,
                    bias=True,
                    num_layers=1,
                    dropout=dropout_W,
                    bidirectional=bidirectional_rnn,
                    batch_first=True
                )
            )
            self.rnn_layer = layers[-1]
            current_dim = args.rnn_dim * (2 if bidirectional_rnn else 1)
        if args.dropout_prob > 0:
            layers.append(nn.Dropout(p=args.dropout_prob, inplace=False))
            self.dropout_layer = layers[-1]
            current_dim = current_dim  # Doesn't change.
        if pooling:
            if args.aggregation == 'mot':
                layers.append(MeanOverTime())
            elif args.aggregation.startswith('att'):
                layers.append(Attention(current_dim, attention_fn=F.tanh, op=args.aggregation))
            else:
                raise NotImplementedError
            self.pooling_layer = layers[-1]
        if args.variety:
            variety_size = 1
            # current_dim += 1

        if args.punct:
            punct_size = 1
            # current_dim += 1
        self.linear = nn.Linear(current_dim, num_outputs)
        layers.append(self.linear)

        if args.variety:
            self.variety_linear = nn.Linear(variety_size, num_outputs)
            layers.append(self.variety_linear)
        if args.punct:
            self.punct_linear = nn.Linear(punct_size, num_outputs)
            layers.append(self.punct_linear)

        if not args.skip_init_bias:
            layers[-1].bias.data = init_bias_value
        self.sigmoid = nn.Sigmoid()
        layers.append(self.sigmoid)
        self.layers = layers
        if args.emb_path:
            logger.info('Initializing lookup table')
            emb_reader = EmbeddingReader(args.emb_path, emb_dim=args.emb_dim)
            layers[0].weight = emb_reader.get_emb_matrix_given_vocab(vocab, layers[0].weight)
            logger.info('  Done')

    def _append_count(self, array, current):
        array = np.array(array)
        temp = torch.unsqueeze(torch.from_numpy(array), 1).float()
        if self.args.cuda:
            temp = temp.cuda()
        part = torch.autograd.Variable(temp, requires_grad=False)
        current = torch.cat((current, part), dim=1)
        return current
    def _append_multiple_counts(self, current, counts):
        '''
            ARGS
            ----
                current: FloatTensor batch_size, cur_dim
                counts: list of (FloatTensor batch_size, count_dim)
        '''
        temp = []
        for count in counts:
            if self.args.cuda:
                temp.append(count.cuda())
            else:
                temp.append(count)
        current = torch.cat([current] + temp, dim=1)  # Have to do one of these.
        if self.args.cuda:
            return current.cuda()
        else:
            return current
    def forward(self, x, mask=None, lens=None, punct=None, variety=None, pos=None):
        '''
            x: Variable, batch_size * max_seq_length
                x is assumed to be padded.
                x should be a LongTensor
            mask: batch_size * max_seq_length
            lens: batch_size LongTensor, lengths of each sequence.
        '''
        if mask is not None and self.args.cuda:
            mask = mask.cuda()
        if lens is not None and self.args.cuda:
            lens = lens.cuda()
        batch_size, max_seq_length = x.size()[0], x.size()[1]
        current = x.long()
        # Embedding
        if self.args.cuda:
            current = current.cuda()
        # pdb.set_trace()
        current = self.embedding_layer(current)
        # current: batch_size * max_seq_length * emb_dim
        if self.args.pos:
            n = pos_dim()
            # size, msl, emb_dim = current.size()
            # one_hot = np.zeros((size, msl, n))
            # for i, j in enumerate(pos):
            #     for ind, elem in enumerate(j):
            #         one_hot[i][ind][elem] = 1
            # ohe = torch.from_numpy(one_hot).float()
            # if self.args.cuda:
            #     ohe = ohe.cuda()
            # var = torch.autograd.Variable(ohe, pos=None, requires_grad=False)
            # var = pos  # TODO
            # Need to do this because pytorch messes with current.size()[1]
            var = pos[:, :current.size()[1], :]
            if self.args.cuda:
                var = var.cuda()
            current = torch.cat((current, var), dim=2)
        if self.args.cuda:
            current = current.cuda()
        # CNN
        if hasattr(self, 'cnn_layer'):
            current = self.cnn_layer(current, mask=mask)
        # RNN
        if hasattr(self, 'rnn_layer'):
            seq_lengths = lens.data.cpu().numpy()
            current = pack_padded_sequence(current,
                                           seq_lengths,
                                           batch_first=True)
            self.rnn_layer.flatten_parameters()
            current, _ = self.rnn_layer(current)  # (h0, c0)
            # current = temp[0]
            current, seq_lengths = pad_packed_sequence(current,
                                                       batch_first=True)
        # Dropout
        if hasattr(self, 'dropout_layer'):
            current = self.dropout_layer(current)
        if self.args.cuda:
            current = current.cuda()
        # Pooling
        if hasattr(self, 'pooling_layer'):
            current = self.pooling_layer(current, mask=mask, lens=lens, dim=1)
        else:
            current = current[lens]

        counts = []
        current = self.linear(current)
        if self.args.variety:
            current += self.variety_linear(variety)
        if self.args.punct:
            current += self.punct_linear(punct)

        current = self.sigmoid(current)
        return current
class EnsembleModel(torch.nn.Module):
    def __init__(self, models, type="mean"):
        super(EnsembleModel, self).__init__()
        models = [torch.load(model, map_location=lambda storage, location: storage) for model in models]
        # Move stuff to CPU and out of DataParallel.
        for i in range(len(models)):
            if type(models[i]) is torch.nn.DataParallel:
                models[i] = models[i].module
            models[i].cpu()
        
        self.models = torch.nn.ModuleList(models)
        self.voting_strategy = type

    def forward(self, x, *args, **kwargs):
        predictions = [model(x, *args, **kwargs) for model in self.models]
        concat_preds = torch.cat(predictions, dim=1)
        if self.voting_strategy == "mean":
            return concat_preds.mean(dim=1)
        elif self.voting_strategy == "median":
            return torch.median(concat_preds, dim=1)
        else:
            raise Exception("Invalid voting strategy")
