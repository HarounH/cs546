'''
    This file implements the a dataset class for ASAP data.
    Instances of the class are then fed into dataloaders
    The dataloaders are used to train/test etc models.
'''

__author__ = 'Haroun'
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
import operator
from collections import defaultdict

from nltk.tag.perceptron import PerceptronTagger
tagger = PerceptronTagger()

POS_DICT = defaultdict(lambda: 0)

POS = [     "CC",     "CD",     "DT",     "EX",     "FW",     "IN",     "JJ",     "JJR",     "JJS",     "LS",     "MD",     "NN",     "NNP",     "NNPS",     "NNS",     "PDT",     "POS",     "PRP",     "PRP$",     "RB",     "RBR",     "RBS",     "RP",     "SYM",     "TO",     "UH",     "VB",     "VBD",     "VBG",     "VBN",     "VBP",     "VBZ",     "WDT",     "WP",     "WP$",     "WRB" ]

PUNCTS = '!?.,;'

for i, pos in enumerate(POS):
    POS_DICT[pos] = i

def pos_dim():
    return max(POS_DICT.values())+1


logger = logging.getLogger(__name__)
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
ref_scores_dtype = 'int32'
if sys.platform in ['win32']:
    print('Detected windows OS')
    tsv_encoding = 'latin-1'
    lineneding = '\n'
else:
    tsv_encoding = 'latin-1'
    lineneding = '\n'


def is_number(string):
    # neater than regex really. stack is annoying but eh
    try:
        lol = float(string)
        return True
    except ValueError:
        return False


# unfortunately, torch.utils.data.Dataset isn't great for NLP
# torchtext is overkill for this. I'm just gonna roll my own.
class ASAPDataset:  # (torch.utils.data.Dataset):
    '''
        Attributes:
            tsv_file
    '''
    asap_ranges = {
        0: (0, 60),
        1: (2, 12),
        2: (1, 6),
        3: (0, 3),
        4: (0, 3),
        5: (0, 4),
        6: (0, 4),
        7: (0, 30),
        8: (0, 60)
    }

    def __init__(self, tsv_file, vocab_size=-1, vocab=None, read_vocab=False, vocab_file=None, prompt_id=-1, pos=False):
        self.tsv_file = tsv_file
        self.prompt_id = prompt_id  # Need this for evaluation.
        if vocab is None:
            if read_vocab is True:
                logging.info('Loading vocab from ' + vocab_file)
                with open(vocab_file, 'rb') as f:
                    self.vocab = pickle.load(f)
            else:
                logger.info('Loading vocab from ' + tsv_file)
                self.vocab = self.create_vocab_from_tsv(tsv_file, pos=pos, vocab_size=vocab_size, prompt_id=prompt_id)
                if vocab_file is not None:
                    logging.info('Writing vocab to ' + vocab_file)
                    with open(vocab_file, 'wb') as f:
                        pickle.dump(self.vocab, f)
        else:
            self.vocab = vocab
        self.unique_x = []
        self.tags_x = []
        self.punct_x = []

        self.ids, self.x, self.y, self.prompts, self.maxlen = \
            self.read_tsv(tsv_file, self.vocab, prompt_id=prompt_id, pos=pos)

        self.prepare_classical_features()

    def __len__(self):
        # Number of essays
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.prompts[idx]

    def tokenize(self, string, pos):
        tokens = nltk.word_tokenize(string)
        for index, token in enumerate(tokens):
            if token == '@' and (index+1) < len(tokens):
                tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
                tokens.pop(index)
        return tokens

    def _tokenize(self, text, pos):
        sentences = nltk.sent_tokenize(text)
        ret = list()
        part_of_speech = list()
        for sentence in sentences:
            tokens = nltk.word_tokenize(sentence)
            if pos:
                tagged = list(map(lambda x: x[-1], tagger.tag(tokens)))
            for index, token in enumerate(tokens):
                if token == '@' and (index+1) < len(tokens):
                    tokens[index+1] = '@' + re.sub('[0-9]+.*', '', tokens[index+1][0])
                    tokens.pop(index)
                    if pos:
                        tagged.pop(index)
                ret.extend(tokens)
            if pos:
                part_of_speech.extend(tagged)
        if pos:
            return ret, part_of_speech
        return ret

    def create_vocab_from_tsv(self, tsv_file, pos=False, vocab_size=-1, maxlen=-1, prompt_id=-1, to_lower=True, tokenize_not_split=True):
        '''
            Reads a tsv_file and constructs a vocabulary (dictionary)
            from that.
            Some indices are reserved.
                0: <pad>
                1: <unk>
                2: <num>
        '''
        total_words, unique_words = 0, 0
        word_freqs = {}
        if maxlen > 0:
            logger.warning('Removing essays with length > ' + str(maxlen))
        with open(tsv_file, 'r', encoding=tsv_encoding) as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count == 1:
                    continue
                line_toks = line.strip('\r\n').split('\t')
                essay_set = int(line_toks[1])
                content = line_toks[2]
                # Do stuff only if for the given prompt
                if essay_set == prompt_id or prompt_id <= 0:
                    if to_lower:
                        content = content.lower()
                    if tokenize_not_split:
                        content = self._tokenize(content, pos)
                    else:
                        content = content.split()
                    if maxlen > 0 and len(content) > maxlen:
                        continue
                    for word in content:
                        try:
                            word_freqs[word] += 1
                        except KeyError:
                            unique_words += 1
                            word_freqs[word] = 1
                        total_words += 1
        logger.info('  %i total words, %i unique words' %
                    (total_words, unique_words))
        import operator
        sorted_word_freqs = sorted(word_freqs.items(),
                                   key=operator.itemgetter(1),
                                   reverse=True)
        if vocab_size <= 0:
            # Choose vocab size automatically by removing all singletons
            vocab_size = 0
            for word, freq in sorted_word_freqs:
                if freq > 1:
                    vocab_size += 1
        vocab = {'<pad>': 0, '<unk>': 1, '<num>': 2}
        vcb_len = len(vocab)
        index = vcb_len
        for freq_rank, (word, freq) in enumerate(sorted_word_freqs):
            if freq_rank < vocab_size - vcb_len:
                vocab[word] = index
                index += 1
            else:
                vocab.pop(word, None)  # Bye bye word
        # pdb.set_trace()
        return vocab

    def read_tsv(self, tsv_file, vocab, char_level=False, tokenize_not_split=True, to_lower=False, maxlen=-1, prompt_id=-1, score_index=6, pos=False):
        logging.info('Reading TSV file from ' + tsv_file)
        if maxlen > 0:
            logger.info('  Removing sequences with more than ' + str(maxlen) +
                        ' words')
        data_ids, data_x, data_y, prompt_ids = [], [], [], []
        num_hit, unk_hit, total = 0., 0., 0.
        maxlen_x = -1
        with open(tsv_file, 'r', encoding=tsv_encoding) as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count == 1:
                    continue  # Skip for header line.
                tokens = line.strip('\r\n').split('\t')
                essay_id = int(tokens[0])
                essay_set = int(tokens[1])
                content = str(tokens[2])
                score = float(tokens[score_index])
                indices = []
                if essay_set == prompt_id or prompt_id < 0:
                    if to_lower:
                        content = content.lower()
                    if char_level:
                        raise NotImplementedError  # TODO
                    else:
                        data = self._tokenize(content, pos)  # Changed from tokenize -> _tokenize
                        if pos:
                            content, tags = data
                            tag_encoded = [POS_DICT[i] for i in tags]
                            self.tags_x.append(tag_encoded)
                        else:
                            content = data
                        for word in content:
                            if is_number(word):
                                indices.append(vocab['<num>'])
                                num_hit += 1
                            elif word in vocab:
                                indices.append(vocab[word])
                            else:
                                indices.append(vocab['<unk>'])
                                unk_hit += 1
                            total += 1
                        data_ids.append(essay_id)
                        data_x.append(indices)
                        self.unique_x.append(len(set(indices)) / len(indices))
                        self.punct_x.append(len([1 for i in content if i in PUNCTS]))
                        data_y.append(score)
                        prompt_ids.append(essay_set)
                        maxlen_x = max(maxlen_x, len(indices))
        self.unique_x = np.array(self.unique_x)
        self.tags_x = np.array(self.tags_x)
        self.punct_x = np.array(self.punct_x)
        logger.info('  <num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
        return data_ids, data_x, data_y, prompt_ids, maxlen_x

    def make_scores_model_friendly(self):
        for i in range(len(self.y)):
            low, high = self.asap_ranges[self.prompts[i]]
            self.y[i] = (self.y[i] - low) / (high - low)

    def make_scores_dataset_friendly(self):
        for i in range(len(self.y)):
            low, high = self.asap_ranges[self.prompts[i]]
            self.y[i] = low + (high - low) * self.y[i]

    def prepare_classical_features(self):
        '''
            Function messes around with how dataset represents
            classical features such as tags_x, unique, punct

            The desired output format is a single
        '''
        # TODO
        # tags_x
        pass

class ASAPDataLoader:
    def __init__(self, dataset, maxlen, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.len = len(dataset)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.len:
            raise StopIteration
        lower = self.index
        self.index += self.batch_size
        higher = min(self.index, self.len)
        xs, ys, prompts = self.dataset[lower:higher]
        lens = []
        batch_max_len = max([len(x) for x in xs])
        for i in range(len(xs)):
            # pdb.set_trace()
            x = xs[i]
            lens.append(len(x))
            x = x + [0 for i in range(batch_max_len - len(x))]
            xs[i] = x
        mask = torch.FloatTensor(
            [
                [1]*lens[i] + [0]*(batch_max_len - lens[i])
                for i in range(len(xs))
                ]
            )
        sorter = np.flip(np.argsort(lens), axis=0).tolist()
        xs = Variable(torch.LongTensor(xs))
        ys = Variable(torch.FloatTensor(ys))
        prompts = Variable(torch.LongTensor(prompts))
        mask = Variable(mask)
        lens = Variable(torch.LongTensor(lens))
        return xs[sorter],\
            ys[sorter],\
            prompts[sorter],\
            mask[sorter],\
            lens[sorter],\
            (lower, higher)


if __name__ == '__main__':
    # This is for testing stuff
    dataset_type = 'train'
    for fold_idx in range(1):
        train_data = ASAPDataset('../data/fold_%d/%s.tsv' % (fold_idx, dataset_type), prompt_id=2)
    # pdb.set_trace()
    print('Loaded')
    for epoch in range(3):
        nbatches = 0
        for (xs, ys, prompts) in ASAPDataLoader(train_data, train_data.maxlen, 20):
            nbatches += 1
            # pdb.set_trace()
        print('Epoch ' + str(epoch) + ' has ' + str(nbatches) + ' batches')
    print('Thing works')
    for epoch in range(3):
        for (xs, ys, prompts) in ASAPDataLoader(train_data, train_data.maxlen, 20):
            pdb.set_trace()
