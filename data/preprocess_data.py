'''
    This file handles forming folds given the input tsv
    Assumes a directory structure similar to https://github.com/nusnlp/nea
    Code is based on https://github.com/nusnlp/nea/blob/master/data/preprocess_asap.py
'''

__author__ = 'Haroun'
__mail__ = 'haroun7@gmail.com'

import os
import sys
import argparse
import pdb

if sys.platform in ['win32']:
    print('Detected windows OS')
    tsv_encoding = 'latin-1'
    lineneding = '\n'
else:
    tsv_encoding = 'latin-1'
    lineneding = '\n'

def collect_dataset(filename):
    data = {}
    header_line = True
    with open(filename, 'r', encoding=tsv_encoding) as f:
        for line in f:
            if header_line:
                data['header'] = line.strip('\r\n')
                header_line = False
                continue
            toks = (line.strip('\r\n')).split('\t')
            data[toks[0]] = line.strip('\r\n')
    return data


def extract_based_on_ids(data, id_file):
    lines = [data['header']]
    with open(id_file, 'r') as f:
        for line in f:
            fixed_line = line.strip('\r\n')
            lines.append(data[(fixed_line).split('\t')[0]])
    return lines


def create_dataset(lines, filename):
    with open(filename, 'w', encoding=tsv_encoding) as f:
        f.write(lineneding.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', dest='input_file', required=True, help='Input TSV file')
    args = parser.parse_args()
    all_lines = collect_dataset(args.input_file)
    for fold_idx in range(0, 5):
        for dataset_type in ['dev', 'test', 'train']:
            lines = extract_based_on_ids(all_lines, 'fold_%d/%s_ids.txt' % (fold_idx, dataset_type))
            create_dataset(lines, 'fold_%d/%s.tsv' % (fold_idx, dataset_type))
