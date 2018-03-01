import numpy as np
import logging
import torch
from torch.nn import Embedding
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .custom_layers import Attention, MeanOverTime, Conv1DWithMasking
from .w2vEmbReader import W2VEmbReader as EmbReader

logger = logging.getLogger(__name__)

dropout_W = 0.5
cnn_border_mode='same'

class Regression(torch.nn.Module):
    """
    """
    def __init__(self, vocab_size: int,
                 embed_dim: int, cnn_dim: int, emb_path: str,
                 rnn_dim: int, pooling: bool, dropout_prob: float,
                 num_outputs: int, skip_init_bias: bool,
                 bias_value: float, aggregation: str, bidirectional: bool,
                 vocab, cnn_window_size: int):
        """
        """
        super(Regression, self).__init__()
        self.embed = Embedding(vocab_size, embed_dim)
        if emb_path:
            logger.info('Initializing lookup table')
            emb_reader = EmbReader(emb_path, emb_dim=embed_dim)
            self.embed.W.data = emb_reader.get_emb_matrix_given_vocab(vocab)

        current_out_dim = embed_dim
        if cnn_dim > 0:
            # We may need to mess with this layer to get the masking right
            self.conv_layer = Conv1DWithMasking(current_out_dim,        # Input Dimension
                        cnn_dim,                # Output Dimension
                        cnn_window_size) # TODO: subsample_length=1 Convert this
            # We also need a kernel for this?
            current_out_dim = cnn_dim

        if rnn_dim > 0:
            self.rnn_layer = \
                torch.nn.RNN(input_size=current_out_dim,
                            hidden_size=rnn_dim,
                            num_layers=1, #???
                            nonlinearity='tanh', #???
                            dropout=dropout_W, #???
                            bidirectional=bidirectional)
            current_out_dim = rnn_dim
            if bidirectional:
                current_out_dim *= 2

        layers = []
        if dropout_prob > 0:
            self.drop_layer = torch.nn.Dropout(p=dropout_prob)

        if pooling:
            if aggregation == 'mot':
                self.agg = MeanOverTime()
            elif aggregation.startswith('att'):
                self.agg = Attention(current_out_dim, op=aggregation, activation='tanh', init_stdev=0.01)
            else:
                raise NotImplementedError("{} not a valid aggregation scheme".format(aggregation))

        ll = torch.nn.Linear(current_out_dim, num_outputs)
        if not skip_init_bias:
            ll.bias.data = torch.from_numpy(np.array([bias_value])).float()
        layers.append(('linear', ll))
        layers.append(('sigmoid', torch.nn.Sigmoid()))
        self.model = torch.nn.Sequential(OrderedDict(layers))
        self.embed_index = 0

    def forward(self, x, mask, one_hot_mask):
        """
        """
        out = self.embed(x).float()
        if hasattr(self, 'conv_layer'):
            out = self.conv_layer(out, mask=one_hot_mask)
        if hasattr(self, 'rnn_layer'):
            np_mask = mask.cpu().numpy()
            packed_input = pack_padded_sequence(out, np_mask)
            packed_output, _ = self.rnn_layer(packed_input)
            out, _ = pad_packed_sequence(packed_output)
            if not hasattr(self, 'agg'):
                # If we don't pool, take the last timestamp
                out = out[-1, :, :].float()

        if hasattr(self, 'dropout'):
            out = self.dropout(out)
        if hasattr(self, 'agg'):
            # We can have pooling without rnns, don't ask me how
            out = self.agg(out, mask=mask)

        return self.model(out)


def create_model(args, num_outputs, initial_mean_value, vocab):
    """

    :param args: Namespace
    :param num_outputs:
    :param initial_mean_value:
    :param vocab:
    :return:
    """

    bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value))

    if args.model_type == 'reg':
        logger.info('Building a REGRESSION model')
        return Regression(args.vocab_size, args.emb_dim,
            args.cnn_dim, args.emb_path,
            args.rnn_dim, False, args.dropout_prob,
            num_outputs, args.skip_init_bias,
            bias_value, args.aggregation, False, vocab,
            args.cnn_window_size)
    elif args.model_type == 'regp':
        logger.info('Building a REGRESSION model with POOLING')
        return Regression(args.vocab_size, args.emb_dim,
            args.cnn_dim, args.emb_path,
            args.rnn_dim, True, args.dropout_prob,
            num_outputs, args.skip_init_bias,
            bias_value, args.aggregation, False, vocab,
            args.cnn_window_size)

    elif args.model_type == 'breg':
        logger.info('Building a BIDIRECTIONAL REGRESSION model')
        return Regression(args.vocab_size, args.emb_dim,
            args.cnn_dim, args.emb_path,
            args.rnn_dim, False, args.dropout_prob,
            num_outputs, args.skip_init_bias,
            bias_value, args.aggregation, True, vocab,
            args.cnn_window_size)
    elif args.model_type == 'bregp':
        logger.info('Building a BIDIRECTIONAL REGRESSION model with POOLING')
        return Regression(args.vocab_size, args.emb_dim,
            args.cnn_dim, args.emb_path,
            args.rnn_dim, True, args.dropout_prob,
            num_outputs, args.skip_init_bias,
            bias_value, args.aggregation, True, vocab,
            args.cnn_window_size)
    raise NotImplementedError('{} not implemented'.format(args.model_type))
