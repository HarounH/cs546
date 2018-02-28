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

def read_dataset(file_path: str, prompt_id: int, maxlen: int, vocab, tokenize_text: bool, 
        to_lower=True :bool, score_index=6 :int) -> pd.Dataframe:

    logger.info('Reading dataset from: {}'.format(file_path))
    if maxlen > 0:
        logger.info('Removing sequences with more than {} words'.format(str(maxlen)))
    data_x, data_y, prompt_ids = [], [], []
    num_hit, unk_hit, total = 0., 0., 0.
    maxlen_x = -1

    # Header is first
    dataset = pd.read_csv(file_path, delimiter="\t", encoding="UTF-8", header=0) # np.loadtxt(file_path, delimiter="\t", encoding="UTF-8", skiprows=0)
    mask = dataset['essay_set'] == prompt_id
    # Select all elements if prompt_id <= 0
    if prompt_id <= 0:
        mask = dataset['essay_id'] == dataset['essay_id']

    subset = dataset[mask]
    if to_lower:
        subset['content'] = subset['content'].str.lower()
    elif tokenize_text:
        subset['content'] = subset['content'].apply(tokenize)
    else:
        subset['content'] = subset['content'].str.split()


    dataset[mask] = subset

    if maxlen < 0:
        maxlen = 9223372036854775807

    subset_mask = subset['content'].apply(lambda x: len(x) <= maxlen)
    


    for line in input_file:
        if essay_set == prompt_id or prompt_id <= 0:


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
    return dataset