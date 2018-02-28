import numpy as np
import sys
import torch
from torch.autograd import Variable
from torch.nn.functional import softmax

class Attention(torch.nn.Module):
    def __init__(self, input_size, op='attsum', activation='tanh', init_stdev=0.01):
        super(Attention, self).__init__()
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        self.att_v = Variable(torch.randn(input_size).mul(self.init_stdev), 
            requires_grad=True)
        self.att_W = Variable(torch.randn(input_size, input_size).mul(self.init_stdev), 
            requires_grad=True)

    def tensordot(self, x, y):
        """
        The last dimension of x and the first dimension of y have to match
        """
        mul = lambda lis: reduce(lambda x, y: x*y, lis, 1)
        x_shape = x.size()
        prev_dims = x_shape[:-1]
        concat_dim = x_shape[-1]
        y_shape = y.size()
        concat_dim_y = y_shape[0]
        after_dims = y_shape[1:]
        assert concat_dim == concat_dim_y
        return torch.mm(x.resize_(mul(prev_dims),concat_dim), 
            y.resize_(concat_dim_y, mul(after_dims))).resize_(prev_dims+after_dims)
    
    def forward(self, x, mask=None):
        y = self.tensordot(x, self.att_W)
        if self.activation == 'tanh':
            y = torch.tanh(y)
        print(self.att_v)
        print(y)
        weights = self.tensordot(y, self.att_v)
        print(weights)
        weights = softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])

        if self.op == 'attsum':
            out = out.sum(axis=0)
        elif self.op == 'attmean':
            out = torch.addcdiv(torch.zeros(mask.size()), out.sum(dim=0), mask)
        return out.float()

class MeanOverTime(torch.nn.Module):
    def __init__(self):
        super(MeanOverTime, self).__init__()

    def forward(self, x, mask=None):
        if mask != None:
            return torch.addcdiv(torch.zeros(mask.size()), x.sum(dim=0), mask)
        else:
            return x.mean(dim=0)

"""
class Attention(Layer):
    def __init__(self, op='attsum', activation='tanh', init_stdev=0.01, **kwargs):
        self.supports_masking = True
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        init_val_v = (np.random.randn(input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_v = K.variable(init_val_v, name='att_v')
        init_val_W = (np.random.randn(input_shape[2], input_shape[2]) * self.init_stdev).astype(K.floatx())
        self.att_W = K.variable(init_val_W, name='att_W')
        self.trainable_weights = [self.att_v, self.att_W]
    
    def call(self, x, mask=None):
        y = K.dot(x, self.att_W)
        if not self.activation:
            weights = K.theano.tensor.tensordot(self.att_v, y, axes=[0, 2])
        elif self.activation == 'tanh':
            weights = K.theano.tensor.tensordot(self.att_v, K.tanh(y), axes=[0, 2])
        weights = K.softmax(weights)
        out = x * K.permute_dimensions(K.repeat(weights, x.shape[2]), [0, 2, 1])
        if self.op == 'attsum':
            out = out.sum(axis=1)
        elif self.op == 'attmean':
            out = out.sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return K.cast(out, K.floatx())

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'op': self.op, 'activation': self.activation, 'init_stdev': self.init_stdev}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class MeanOverTime(Layer):
    def __init__(self, mask_zero=True, **kwargs):
        self.mask_zero = mask_zero
        self.supports_masking = True
        super(MeanOverTime, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if self.mask_zero:
            return K.cast(x.sum(axis=1) / mask.sum(axis=1, keepdims=True), K.floatx())
        else:
            return K.mean(x, axis=1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2])
    
    def compute_mask(self, x, mask):
        return None
    
    def get_config(self):
        config = {'mask_zero': self.mask_zero}
        base_config = super(MeanOverTime, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
class Conv1DWithMasking(Convolution1D):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Conv1DWithMasking, self).__init__(**kwargs)
    
    def compute_mask(self, x, mask):
        return mask
"""