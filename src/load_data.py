import logging
import pandas as pd 
import nltk
from typing import List

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def tokenize(string: str) -> List[str]:
    """
        Takes a string and tokenizes the words using nltk tokenizer
        Also merges an element and its adjacent element if the first
        element is an @ (the dataset recognizes those as named entities)
    """

    tokens = nltk.word_tokenize(string)
    ret = []
    for index, token in enumerate(tokens):
        if token == '@' and (index+1) < len(tokens):
            token = '@' + re.sub('[0-9]+.*', '', tokens[index+1])
            tokens.pop(index)
        ret.append(token)
    return ret

def read_dataset(file_path, prompt_id, maxlen, vocab, tokenize_text, 
        to_lower=None, score_index=6):

    logger.info('Reading dataset from: {}'.format(file_path))
    if maxlen > 0:
        logger.info('Removing sequences with more than {} words'.format(str(maxlen)))
    data_x, data_y, prompt_ids = [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = -1

    dataset = pd.read_csv(file_path, sep="\t", encoding="UTF-8", header=0)

    with codecs.open(file_path, mode='r', encoding='UTF8') as input_file:
        input_file.next()
        for line in input_file:
            if essay_set == prompt_id or prompt_id <= 0:
                if to_lower:
                    content = content.lower()
                else:
                    if tokenize_text:
                        content = tokenize(content)
                    else:
                        content = content.split()
                if maxlen > 0 and len(content) > maxlen:
                    continue
                indices = []
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
                data_x.append(indices)
                data_y.append(score)
                prompt_ids.append(essay_set)
                if maxlen_x < len(indices):
                    maxlen_x = len(indices)
    logger.info('<num> hit rate: %.2f%%, <unk> hit rate: %.2f%%' % (100*num_hit/total, 100*unk_hit/total))
    return data_x, data_y, prompt_ids, maxlen_x