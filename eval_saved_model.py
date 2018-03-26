import argparse
import torch
from src.dataset import ASAPDataset, ASAPDataLoader
import numpy as np
from src.qwk import quadratic_kappa

def main(args):
    if not hasattr(args, 'out_dir'):
        args.out_dir = "out/"
    model = torch.load(args.model, map_location=lambda storage, location: storage)
    model.cpu()
    # train
    train_dataset = ASAPDataset(args.train_path, vocab_file=args.out_dir + '/vocab.pkl', pos=args.pos)
    vocab = train_dataset.vocab
    train_dataset.make_scores_model_friendly()
    # test
    test_dataset = ASAPDataset(args.test_path, vocab=vocab, pos=args.pos)
    test_dataset.make_scores_model_friendly()
    # dev
    dev_dataset = ASAPDataset(args.dev_path, vocab=vocab, pos=args.pos)
    dev_dataset.make_scores_model_friendly()

    loader = ASAPDataLoader(test_dataset, train_dataset.maxlen, 9999999999999)
    for xs, ys, ps, padding_mask, lens, bounds in loader:
        pred = model(xs, mask=padding_mask, lens=lens)
        print("Quadratic kappa: {}".format(quadratic_kappa(pred, ys)))

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Evaluates a saved model')
    parse.add_argument('-m', '--model', required=True, type=str, metavar='<str>',
                    help='Model path')
    parse.add_argument('-r', '--train', dest="train_path", required=True, type=str, metavar='<str>',
                    help='Path to the training dataset (needed for vocabs)')
    parse.add_argument('-t', '--test-path', dest="test_path" , required=True, type=str, metavar='<str>',
                    help='Path to the test dataset')
    parse.add_argument('-d', '--dev-path', dest="dev_path", required=True, type=str, metavar='<str>',
                    help='Path to the development ids')
    parse.add_argument('-p', '--pos', dest="pos", action="store_true",
                    help='Whether to use POS in the model (the model must be trained with pos)')

    args = parse.parse_args()
    main(args)