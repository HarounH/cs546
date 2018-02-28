import numpy as np
import logging
import torch
from torch.nn import Embedding
from collections import OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)

dropout_W = 0.5
dropout_U = 0.1
cnn_border_mode='same'

class UnidirectionalRegression(torch.nn.Module):

    def __init__(self, vocab_size: int, 
            embed_dim: int, cnn_dim: int, emb_path: str, 
            rnn_dim: int, pooling: bool, dropout_prob: float, 
            num_outputs: int, skip_init_bias: bool,
            bias_value: float, aggregation: str):
        """
        """
        super(UnidirectionalRegression, self).__init__()

        self.embed = Embedding(vocab_size, embed_dim)
        current_out_dim = embed_dim
        if cnn_dim > 0:
            self.conv_layer = torch.nn.Conv1d(current_out_dim,        # Input Dimension
                        cnn_dim,                # Output Dimension
                        stride=cnn_window_size) # TODO: subsample_length=1 Convert this
            # We also need a kernel for this?
            current_out_dim = cnn_dim

        if rnn_dim > 0:
            self.rnn_layer = torch.nn.RNN(input_size=current_out_dim, 
                hidden_size=rnn_dim,
                num_layers=1, #???
                nonlinearity='tanh', #???
                dropout=dropout_W #???
                )
                #return_sequences=pooling)
            current_out_dim = rnn_dim

        layers = []
        if dropout_prob > 0:
            drop_layer = torch.nn.Dropout(p=dropout_prob)
            layers.append(('drop_layer', drop_layer))

        self.pooling = pooling
        if pooling:
            if aggregation == 'mot':
                pass
                # layers.append(MeanOverTime(mask_zero=True))
            elif aggregation.startswith('att'):
                pass
                # layers.append(Attention(op=aggregation, activation='tanh', init_stdev=0.01))

        ll = torch.nn.Linear(current_out_dim, num_outputs)
        if not skip_init_bias:
            ll.bias.data = torch.from_numpy(np.array([bias_value])).float()
        layers.append(('linear', ll))
        layers.append(('sigmoid', torch.nn.Sigmoid()))
        self.model = torch.nn.Sequential(OrderedDict(layers))
        self.embed_index = 0

    def forward(self, x, mask: ):
        """
        """
        out = self.embed(x).float()
        if hasattr(self, 'conv_layer'):
            out = self.conv_layer(out)
        if hasattr(self, 'rnn_layer'):
            packed_input = pack_padded_sequence(out, mask.cpu().numpy())
            packed_output, _ = self.rnn_layer(packed_input)
            out, _ = pad_packed_sequence(packed_output)
            if self.pooling:
                # Train with dropout over time
                # Aggregate with MeanOverTime or Attention
                # Feed to LinearLayer
            else:
                out = out[-1, :, :].float()

        return self.model(out)

class BidirectionalRegression(torch.nn.Module):

    def __init__(self, vocab_size, 
            embed_dim, cnn_dim, emb_path, rnn_dim, pooling: bool):
        """
        """
        super(BidirectionalRegression, self).__init__()
        self.embedding = Embedding(vocab_size, embed_dim) 


    def forward(self, x):
        """
        """
        return x

def create_model(args, num_outputs, initial_mean_value, overal_maxlen, vocab):
    bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value))
    if args.model_type == 'reg':
        logger.info('Building a REGRESSION model')
        return UnidirectionalRegression(args.vocab_size, args.emb_dim, 
            args.cnn_dim, args.emb_path, 
            args.rnn_dim, False, args.dropout_prob, 
            num_outputs, args.skip_init_bias, 
            bias_value, args.aggregation
            )
    elif args.model_type == 'regp':
        logger.info('Building a REGRESSION model with POOLING')
        return UnidirectionalRegression(args.vocab_size, args.emb_dim, 
            args.cnn_dim, args.emb_path, 
            args.rnn_dim, True, args.dropout_prob, 
            num_outputs, args.skip_init_bias, 
            bias_value, args.aggregation
            )
        """
        model = Sequential()
        model.add(Embedding(args.vocab_size, args.emb_dim, mask_zero=True))
        if args.cnn_dim > 0:
            model.add(Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1))
        if args.rnn_dim > 0:
            model.add(RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U))
        if args.dropout_prob > 0:
            model.add(Dropout(args.dropout_prob))

        if args.aggregation == 'mot':
            model.add(MeanOverTime(mask_zero=True))
        elif args.aggregation.startswith('att'):
            model.add(Attention(op=args.aggregation, activation='tanh', init_stdev=0.01))

        model.add(Dense(num_outputs))
        if not args.skip_init_bias:
            bias_value = (np.log(initial_mean_value) - np.log(1 - initial_mean_value)).astype(K.floatx())
            model.layers[-1].b.set_value(bias_value)
        model.add(Activation('sigmoid'))
        model.emb_index = 0
        """
    
    elif args.model_type == 'breg':
        logger.info('Building a BIDIRECTIONAL REGRESSION model')
        model = Sequential()
        sequence = Input(shape=(overal_maxlen,), dtype='int32')
        output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
        if args.cnn_dim > 0:
            output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
        if args.rnn_dim > 0:
            forwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U)(output)
            backwards = RNN(args.rnn_dim, return_sequences=False, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
        if args.dropout_prob > 0:
            forwards = Dropout(args.dropout_prob)(forwards)
            backwards = Dropout(args.dropout_prob)(backwards)
        merged = merge([forwards, backwards], mode='concat', concat_axis=-1)
        densed = Dense(num_outputs)(merged)
        if not args.skip_init_bias:
            raise NotImplementedError
        score = Activation('sigmoid')(densed)
        model = Model(input=sequence, output=score)
        model.emb_index = 1
    
    elif args.model_type == 'bregp':
        logger.info('Building a BIDIRECTIONAL REGRESSION model with POOLING')
        from keras.layers import Dense, Dropout, Embedding, LSTM, Input, merge
        model = Sequential()
        sequence = Input(shape=(overal_maxlen,), dtype='int32')
        output = Embedding(args.vocab_size, args.emb_dim, mask_zero=True)(sequence)
        if args.cnn_dim > 0:
            output = Conv1DWithMasking(nb_filter=args.cnn_dim, filter_length=args.cnn_window_size, border_mode=cnn_border_mode, subsample_length=1)(output)
        if args.rnn_dim > 0:
            forwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U)(output)
            backwards = RNN(args.rnn_dim, return_sequences=True, dropout_W=dropout_W, dropout_U=dropout_U, go_backwards=True)(output)
        if args.dropout_prob > 0:
            forwards = Dropout(args.dropout_prob)(forwards)
            backwards = Dropout(args.dropout_prob)(backwards)
        forwards_mean = MeanOverTime(mask_zero=True)(forwards)
        backwards_mean = MeanOverTime(mask_zero=True)(backwards)
        merged = merge([forwards_mean, backwards_mean], mode='concat', concat_axis=-1)
        densed = Dense(num_outputs)(merged)
        if not args.skip_init_bias:
            raise NotImplementedError
        score = Activation('sigmoid')(densed)
        model = Model(input=sequence, output=score)
        model.emb_index = 1
    
    logger.info('  Done')
    
    ###############################################################################################################################
    ## Initialize embeddings if requested
    #

    if args.emb_path:
        from w2vEmbReader import W2VEmbReader as EmbReader
        logger.info('Initializing lookup table')
        emb_reader = EmbReader(args.emb_path, emb_dim=args.emb_dim)
        model.layers[model.emb_index].W.set_value(emb_reader.get_emb_matrix_given_vocab(vocab))
        logger.info('  Done')
    
    return model
