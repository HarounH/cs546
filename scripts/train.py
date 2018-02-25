#!/usr/bin/env python3.6

import argparse
import logging
import numpy as np
import scipy
from time import time
import sys
import os
import pickle as pk
#from src.asap_evaluator import Evaluator
#import src.asap_reader as dataset
from jsonschema import validate
import yaml
import pathlib
from pprint import pprint, pformat

# To refactor
#from keras.preprocessing import sequence

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def parse_arguments():
    """
        Parses the config file and returns a dictionary
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", dest="config_file", type=str, 
        metavar='<str>', required=True, help="The path configuration file")
    parser.add_argument("-s", "--schema", dest="schema_file", type=str, 
        metavar='<str>', required=True, default="schema.yaml", help="The path configuration file")
    args = parser.parse_args()
    return args

def validate_params(config_file: str, schema_file: str) -> dict:
    """
        Validates all the params int the config fields
    """
    config_file = yaml.load(open(config_file).read())
    schema_file = yaml.load(open(schema_file).read())
    validate(config_file, schema_file)
    return config_file

def configure_logger(output_directory: str) -> None:
    """
        Configures the logger and their format
    """
    global logger
    pathlib.Path(output_directory).mkdir(parents=True, exist_ok=True) 
    file_format = '[%(levelname)s] (%(name)s) %(message)s'
    log_filename = os.path.splitext(os.path.basename(__file__))[0]
    log_file = logging.FileHandler(os.path.join(output_directory, 
        '{}_log.txt'.format(log_filename)), mode='w')
    log_file.setLevel(logging.DEBUG)
    log_file.setFormatter(logging.Formatter(file_format))
    logger.addHandler(log_file)

def main() -> None:
    """
    """
    args = parse_arguments()
    configs = validate_params(args.config_file, args.schema_file)

    np.random.seed(configs['seed'])
    configure_logger(configs['out_dir_path'])
    logger.debug(pformat(configs))

if __name__ == '__main__':
    main()


"""

###############################################################################################################################
## Prepare data
#

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
    train_x = sequence.pad_sequences(train_x, maxlen=overal_maxlen)
    dev_x = sequence.pad_sequences(dev_x, maxlen=overal_maxlen)
    test_x = sequence.pad_sequences(test_x, maxlen=overal_maxlen)
else:
    train_x = sequence.pad_sequences(train_x)
    dev_x = sequence.pad_sequences(dev_x)
    test_x = sequence.pad_sequences(test_x)

###############################################################################################################################
## Some statistics
#

import keras.backend as K

train_y = np.array(train_y, dtype=K.floatx())
dev_y = np.array(dev_y, dtype=K.floatx())
test_y = np.array(test_y, dtype=K.floatx())

if args.prompt_id:
    train_pmt = np.array(train_pmt, dtype='int32')
    dev_pmt = np.array(dev_pmt, dtype='int32')
    test_pmt = np.array(test_pmt, dtype='int32')

bincounts, mfs_list = utils.bincounts(train_y)
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

logger.info('  train_x shape: ' + str(np.array(train_x).shape))
logger.info('  dev_x shape:   ' + str(np.array(dev_x).shape))
logger.info('  test_x shape:  ' + str(np.array(test_x).shape))

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
## Optimizaer algorithm
#

from nea.optimizers import get_optimizer

optimizer = get_optimizer(args)

###############################################################################################################################
## Building model
#

from nea.models import create_model

if args.loss == 'mse':
    loss = 'mean_squared_error'
    metric = 'mean_absolute_error'
else:
    loss = 'mean_absolute_error'
    metric = 'mean_squared_error'

model = create_model(args, train_y.mean(axis=0), overal_maxlen, vocab)
model.compile(loss=loss, optimizer=optimizer, metrics=[metric])

###############################################################################################################################
## Plotting model
#

from keras.utils.visualize_util import plot

plot(model, to_file = out_dir + '/model.png')

###############################################################################################################################
## Save model architecture
#

logger.info('Saving model architecture')
with open(out_dir + '/model_arch.json', 'w') as arch:
    arch.write(model.to_json(indent=2))
logger.info('  Done')
    
###############################################################################################################################
## Evaluator
#

evl = Evaluator(dataset, args.prompt_id, out_dir, dev_x, test_x, dev_y, test_y, dev_y_org, test_y_org)

###############################################################################################################################
## Training
#

logger.info('--------------------------------------------------------------------------------------------------------------------------')
logger.info('Initial Evaluation:')
evl.evaluate(model, -1, print_info=True)

total_train_time = 0
total_eval_time = 0

for ii in range(args.epochs):
    # Training
    t0 = time()
    train_history = model.fit(train_x, train_y, batch_size=args.batch_size, nb_epoch=1, verbose=0)
    tr_time = time() - t0
    total_train_time += tr_time
    
    # Evaluate
    t0 = time()
    evl.evaluate(model, ii)
    evl_time = time() - t0
    total_eval_time += evl_time
    
    # Print information
    train_loss = train_history.history['loss'][0]
    train_metric = train_history.history[metric][0]
    logger.info('Epoch %d, train: %is, evaluation: %is' % (ii, tr_time, evl_time))
    logger.info('[Train] loss: %.4f, metric: %.4f' % (train_loss, train_metric))
    evl.print_info()

###############################################################################################################################
## Summary of the results
#

logger.info('Training:   %i seconds in total' % total_train_time)
logger.info('Evaluation: %i seconds in total' % total_eval_time)

evl.print_final_info()

if __name__ == '__main__':
    # main()

"""