'''
    File to get data for AES
'''
__author__ = 'Haroun Habeeb'
__mail__ = 'haroun7@gmail.com'

import os
import pdb
import urllib.request
# Grabs ASAP data from https://github.com/nusnlp/nea/tree/master/data
# The data itself is 5 fold, and each fold has a 60-20-20 (?) split.
# This script downloads those individual files
github_prefix = 'https://raw.githubusercontent.com/nusnlp/nea/master/data/'
n_folds = 5
filenames = ['dev_ids.txt', 'test_ids.txt', 'train_ids.txt']
# Get each fold seperately
for fold_idx in range(n_folds):
    # make directory.
    dir_name = 'fold_' + str(fold_idx)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    for filename in filenames:
        url = github_prefix + dir_name + '/' + filename
        destination = dir_name + '/' + filename
        urllib.request.urlretrieve(url, destination)
