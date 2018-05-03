'''
    File implements a general purpose functions
'''

__author__ = 'haroun habeeb'
__mail__ = 'haroun7@gmail.com'

# general imports
import logging
import argparse
import pickle
import os
import sys
import errno
import pdb
import numpy as np
import logging
import nltk
import re
# pytorch imports
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Bernoulli


def tensordot(x, y):
    '''
        x, y are Tensors
            x.Size()[-1] == y.Size()[0]
            multiplicy the last dim of x by the first dimension of y
        x, y should not both be 1 dimensional
    '''
    outshape = x.size()[:-1] + y.size()[1:]
    common_dim = x.size()[-1]
    # pdb.set_trace()
    return x.contiguous().view(-1, common_dim).mm(y.view(common_dim, -1)).view(outshape)


def set_logger(out_dir=None):
    console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
    # datefmt='%Y-%m-%d %Hh-%Mm-%Ss'
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(logging.Formatter(console_format))
    logger.addHandler(console)
    if out_dir:
        file_format = '[%(levelname)s] (%(name)s) %(message)s'
        log_file = logging.FileHandler(out_dir + '/log.txt', mode='w')
        log_file.setLevel(logging.DEBUG)
        log_file.setFormatter(logging.Formatter(file_format))
        logger.addHandler(log_file)


def mkdir_p(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_root_dir():
    return os.path.dirname(sys.argv[0])


def bincounts(array):
    num_rows = array.shape[0]
    if array.ndim > 1:
        num_cols = array.shape[1]
    else:
        num_cols = 1
        array = array[:, None]
    counters = []
    mfe_list = []
    for col in range(num_cols):
        counter = {}
        for row in range(num_rows):
            element = array[row, col]
            if element in counter:
                counter[element] += 1
            else:
                counter[element] = 1
        max_count = 0
        for element in counter:
            if counter[element] > max_count:
                max_count = counter[element]
                mfe = element
        counters.append(counter)
        mfe_list.append(mfe)
    return counters, mfe_list


# Convert all arguments to strings
def ltos(*args):
    outputs = []
    for arg in args:
        if type(arg) == list:
            out = ' '.join(['%.3f' % e for e in arg])
            if len(arg) == 1:
                outputs.append(out)
            else:
                outputs.append('[' + out + ']')
        else:
            outputs.append(str(arg))
    return tuple(outputs)


class BColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    WHITE = '\033[37m'
    YELLOW = '\033[33m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    CYAN = '\033[36m'
    RED = '\033[31m'
    MAGENTA = '\033[35m'
    BLACK = '\033[30m'
    BHEADER = BOLD + '\033[95m'
    BOKBLUE = BOLD + '\033[94m'
    BOKGREEN = BOLD + '\033[92m'
    BWARNING = BOLD + '\033[93m'
    BFAIL = BOLD + '\033[91m'
    BUNDERLINE = BOLD + '\033[4m'
    BWHITE = BOLD + '\033[37m'
    BYELLOW = BOLD + '\033[33m'
    BGREEN = BOLD + '\033[32m'
    BBLUE = BOLD + '\033[34m'
    BCYAN = BOLD + '\033[36m'
    BRED = BOLD + '\033[31m'
    BMAGENTA = BOLD + '\033[35m'
    BBLACK = BOLD + '\033[30m'

    @staticmethod
    def cleared(s):
        return re.sub("\033\[[0-9][0-9]?m", "", s)


def red(message):
    return BColors.RED + str(message) + BColors.ENDC


def b_red(message):
    return BColors.BRED + str(message) + BColors.ENDC


def blue(message):
    return BColors.BLUE + str(message) + BColors.ENDC


def b_yellow(message):
    return BColors.BYELLOW + str(message) + BColors.ENDC


def green(message):
    return BColors.GREEN + str(message) + BColors.ENDC


def b_green(message):
    return BColors.BGREEN + str(message) + BColors.ENDC


def print_args(args, path=None):
    if path:
        output_file = open(path, 'w')
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    args.command = ' '.join(sys.argv)
    items = vars(args)
    for key in sorted(items.keys(), key=lambda s: s.lower()):
        value = items[key]
        if not value:
            value = "None"
        logger.info("  " + key + ": " + str(items[key]))
        if path is not None:
            output_file.write("  " + key + ": " + str(items[key]) + "\n")
    if path:
        output_file.close()
    del args.command


def get_optimizer(args, parameters):
    clipvalue = 0
    clipnorm = 10

    if args.algorithm == 'rmsprop':
        # opt.RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
        optimizer = optim.RMSprop(parameters, lr=0.001, eps=1e-06, alpha=0.9)
    elif args.algorithm == 'sgd':
        # opt.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False, clipnorm=clipnorm, clipvalue=clipvalue)
        optimizer = optim.SGD(parameters, lr=0.01)
    elif args.algorithm == 'adagrad':
        # opt.Adagrad(lr=0.01, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
        optimizer = optim.Adagrad(parameters, lr=0.01)
    elif args.algorithm == 'adadelta':
        # opt.Adadelta(lr=1.0, rho=0.95, epsilon=1e-06, clipnorm=clipnorm, clipvalue=clipvalue)
        optimizer = optim.Adadelta(parameters, lr=1.0, rho=0.95)
    elif args.algorithm == 'adam':
        # opt.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
        optimizer = optim.Adam(parameters, lr=0.001, betas=(0.9, 0.999))
    elif args.algorithm == 'adamax':
        # opt.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=clipnorm, clipvalue=clipvalue)
        optimizer = optim.Adamax(parameters, lr=0.002, betas=(0.9, 0.999))
    return optimizer
