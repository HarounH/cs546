import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
import functools


class Attention(torch.nn.Module):
    """
    Standard attention pooling nlp layer with masking
    """
    def __init__(self, input_size: int, op: str='attsum',
                 activation: str='tanh', init_stdev: float=0.01):
        """
        Initializes parameters

        :param input_size: Input size to layer
        :param op: Either 'attsum' or 'attmean'
        :param activation: Either 'tanh' or None
        :param init_stdev: initial standard deviation
        """
        super(Attention, self).__init__()
        assert op in {'attsum', 'attmean'}
        assert activation in {None, 'tanh'}
        self.op = op
        self.activation = activation
        self.init_stdev = init_stdev
        self.input_size = input_size
        self.att_v = Variable(torch.randn(input_size).mul(self.init_stdev),
                              requires_grad=True)
        self.att_W = Variable(torch.randn(input_size, input_size).mul(self.init_stdev),
                              requires_grad=True)

    @staticmethod
    def tensordot(x: torch.FloatTensor, y: torch.FloatTensor):
        """
        Takes two tensors of size
        X.size() = (a, b, ..., n)
        Y.size() = (n, x, ..., z)
        And returns a Tensor by multiplying the last index of X
        and the first index of Y
        """
        assert x.size()[-1] == y.size()[0]
        if len(x.size()) == 1:
            x = x.view(1, x.size()[0])
        if len(y.size()) == 1:
            y = y.view(1, y.size()[0])

        def mul(lis):
            return functools.reduct(lambda a, b: a*b, lis, 1)

        x_shape = x.size()
        prev_dims = x_shape[:-1]
        concat_dim = x_shape[-1]
        y_shape = y.size()
        concat_dim_y = y_shape[0]
        after_dims = y_shape[1:]
        assert concat_dim == concat_dim_y

        return (x.view(mul(prev_dims), concat_dim).mm(y)).view(prev_dims + after_dims)
    
    def forward(self, x: torch.FloatTensor, mask=None):
        """
        Standard forward pass
        :param x: input
        :param mask: Tensor of length of relevant entries
        :return: The computed layer
        """

        y = Attention.tensordot(x, self.att_W)
        if self.activation == 'tanh':
            y = torch.tanh(y)
        # Reshape so that tensordot is happy
        weights = self.tensordot(y, self.att_v.view((self.input_size, 1)))
        # Softmax needs the attention dimension
        weights = softmax(weights, dim=2)

        # Re-multiply the input
        repeated = torch.cat([weights] * self.input_size, dim=2)
        out = x * repeated

        # no need to incorporate the mask because 0 is the padding
        out = out.sum(dim=0)
        if self.op == 'attmean':
            # If taking the mean, we should only divide over the non-zero activations
            # So we divide each entry by the number of entries
            adjusted_mask = torch.cat([mask.view(mask.size()[0], 1)] * x.shape[-1], dim=0).float()
            out = out / Variable(adjusted_mask, requires_grad=False)
        return out.float()


class MeanOverTime(torch.nn.Module):
    """
    Calculates the (possible masked) mean over time. Time being
    the number word int the sentence
    """
    def __init__(self):
        """
        Nothing to see here
        """
        super(MeanOverTime, self).__init__()

    def forward(self, x: torch.FloatTensor, mask=None):
        """

        :param x: input
        :param mask: A tensor with a list of relevant lengths
        :return: The result of the MeanOverTime
        """
        if isinstance(mask, torch.LongTensor):
            # Divide each element with the number of relevant entries
            # We can do this because all the other entries are set to 0
            # Meaning that they won't contribute to the gradient
            # We may need a straight masking layer as well
            adjusted_mask = torch.cat([mask.view(mask.size()[0], 1)] * x.shape[-1], dim=1).float()
            return x.sum(dim=0) / Variable(adjusted_mask, requires_grad=False)
        else:
            return x.mean(dim=0)


class Conv1DWithMasking(torch.nn.Module):
    """
    1 Dimensional Convolution where the padding is already added
    to the input. Has optional masking
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the light wrapper
        :param args: args to torch.nn.Conv1d
        :param kwargs: kwargs to torch.nn.Conv1d
        """

        super(Conv1DWithMasking, self).__init__()
        self.model = torch.nn.Conv1d(*args, **kwargs)
    
    def forward(self, x: torch.FloatTensor, mask=None):
        """

        :param x: input layer
        :param mask: a 2d tensor that is set to 1 if the element is not to be
            masked out and 0 otherwise.
        :return: The convoluted and masked vector
        """

        x = self.model(x)
        if isinstance(mask, torch.LongTensor):
            x = x * mask
        return x
