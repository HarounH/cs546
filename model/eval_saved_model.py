import argparse
import torch
from src.dataset import ASAPDataset, ASAPDataLoader
import numpy as np
from src.qwk import quadratic_kappa
import pdb


def main(args):
    if not hasattr(args, 'out_dir'):
        args.out_dir = "output_dir/"
    prompt = args.prompt
    model = torch.load(args.model, map_location=lambda storage, location: storage)
    if args.dataparallel:
        model = model.module
    model.cpu()
    model.args.cuda = False
    # model.cpu()
    # train
    train_dataset = ASAPDataset(args.train_path, vocab_file=args.out_dir + '/vocab.pkl', pos=args.pos, prompt_id=args.prompt, maxlen=args.maxlen, vocab_size=args.vocab_size)
    vocab = train_dataset.vocab
    train_dataset.make_scores_model_friendly()
    # test
    test_dataset = ASAPDataset(args.test_path, vocab=vocab, pos=args.pos, prompt_id=args.prompt, maxlen=args.maxlen, vocab_size=args.vocab_size)
    test_dataset.make_scores_model_friendly()
    # dev
    dev_dataset = ASAPDataset(args.dev_path, vocab=vocab, pos=args.pos, prompt_id=args.prompt, maxlen=args.maxlen, vocab_size=args.vocab_size)
    dev_dataset.make_scores_model_friendly()

    lhs, rhs = ASAPDataset.asap_ranges[args.prompt]
    num_ratings = rhs - lhs + 1
    loader = ASAPDataLoader(test_dataset, train_dataset.maxlen, args.batch_size)
    true_ys = []
    pred_ys = []
    #pdb.set_trace()
    batch = -1
    for xs, ys, ps, padding_mask, lens, bounds in loader:
        batch += 1
        print('Starting batch', batch)
        xs.cpu()
        ys.cpu()
        #pdb.set_trace()
        pred = model(xs, mask=padding_mask, lens=lens)
        #pdb.set_trace()
        true_ys.append(ys)
        pred_ys.append(pred.detach().squeeze().data)
        #pdb.set_trace()
    #pdb.set_trace()
    true_ys = torch.stack(true_ys, dim=0)
    pred_ys = torch.stack(pred_ys, dim=0)
    print("Quadratic kappa: {}".format(quadratic_kappa(pred_ys, true_ys, lhs, rhs)))

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
    parse.add_argument('--prompt', dest="prompt", type=int, required=True,
                    help='Prompt id')
    # Maxlen and vocab size
    parse.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parse.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
    parse.add_argument('--dataparallel', type=bool, default=True, help='(Set to true if saved model was a DataParallel model')
    parse.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size to use for testing. CANT BUY MOAR RAM')
    args = parse.parse_args()

    main(args)
