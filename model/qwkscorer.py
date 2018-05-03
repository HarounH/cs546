import argparse
import torch
from src.dataset import ASAPDataset, ASAPDataLoader
import numpy as np
from src.qwk import quadratic_weighted_kappa
import pdb
import os
import torch.nn

    # train
    train_dataset = ASAPDataset(args.train_path, vocab_file=args.out_dir + '/vocab.pkl', pos=args.pos, prompt_id=args.prompt, maxlen=args.maxlen, vocab_size=args.vocab_size)
    vocab = train_dataset.vocab
    # scores are already dataset friendly
    # test
    test_dataset = ASAPDataset(args.test_path, vocab=vocab, pos=args.pos, prompt_id=args.prompt, maxlen=args.maxlen, vocab_size=args.vocab_size)
    # Scores are already dataset friendly
    # dev
    dev_dataset = ASAPDataset(args.dev_path, vocab=vocab, pos=args.pos, prompt_id=args.prompt, maxlen=args.maxlen, vocab_size=args.vocab_size)
    # Scores are already dataset friendly

    score =[]
    files = os.listdir(args.model)
    for file in files:
        path = os.path.join(args.model,file)
        print("processing this file:" + path)
        model = torch.load(path, map_location=lambda storage, location: storage)
        model.eval()
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        if args.cuda:
            model.cuda()
        else:
            print('in cpu')
            model.cpu()
        if isinstance(model, torch.nn.DataParallel) and not(args.cuda):
            model.args.cuda = False
        # model.cpu()

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
            if args.pos:
                indexes = test_dataset.tags_x[bounds[0]:bounds[1]]
            else:
                indexes = None
            if args.variety:
                variety = test_dataset.unique_x[bounds[0]:bounds[1]]
                if args.cuda:
                    variety = variety.cuda()
                else:
                    variety = variety.cpu()
            else:
                variety = None
            if args.punct:
                punct = test_dataset.punct_x[bounds[0]:bounds[1]]
                if args.cuda:
                    punct = punct.cuda()
                else:
                    punct = punct.cpu()
            else:
                punct = None

            pred = model(xs,
                         mask=padding_mask,
                         lens=lens,
                         pos=indexes,
                         variety=variety,
                         punct=punct)
            #pdb.set_trace()
            true_ys.append(ys.data)
            pred_ys.append(pred.detach().squeeze().data)
            #pdb.set_trace()
        #pdb.set_trace()

        true_ys = torch.cat(true_ys).cpu().numpy().squeeze()
        pred_ys = np.rint(torch.cat(pred_ys).cpu().numpy().squeeze() * (rhs - lhs) + lhs)
        score.append(quadratic_weighted_kappa(pred_ys, true_ys, min_rating=lhs, max_rating=rhs))
        print("Quadratic kappa: {}".format(quadratic_weighted_kappa(pred_ys, true_ys, min_rating=lhs, max_rating=rhs)))

    sortedscore =sorted(score, reverse=True)
    for count in range(min(10,len(score))):
        val = score.index(sortedscore[count])
        print('Rank : %d Score: %f : Name: %s'%(count,sortedscore[count], files[val]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates a saved model')
    parser.add_argument('-m', '--model', required=True, type=str, metavar='<str>',
                    help='Model path')
    parser.add_argument('-r', '--train', dest="train_path", required=True, type=str, metavar='<str>',
                    help='Path to the training dataset (needed for vocabs)')
    parser.add_argument('-t', '--test-path', dest="test_path" , required=True, type=str, metavar='<str>',
                    help='Path to the test dataset')
    parser.add_argument('-d', '--dev-path', dest="dev_path", required=True, type=str, metavar='<str>',
                    help='Path to the development ids')
    #parser.add_argument('-p', '--pos', dest="pos", action="store_true",
    #                help='Whether to use POS in the model (the model must be trained with pos)')
    parser.add_argument('--prompt', dest="prompt", type=int, required=True,
                    help='Prompt id')
    # Maxlen and vocab size
    parser.add_argument("--maxlen", dest="maxlen", type=int, metavar='<int>', default=0, help="Maximum allowed number of words during training. '0' means no limit (default=0)")
    parser.add_argument("-v", "--vocab-size", dest="vocab_size", type=int, metavar='<int>', default=4000, help="Vocab size (default=4000)")
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch size to use for testing. CANT BUY MOAR RAM')
    parser.add_argument("--pos", dest="pos", action='store_true', help="Use part of speech tagging in the training")
    parser.add_argument("--variety", dest="variety", action='store_true', help="Variety of words in output layer")
    parser.add_argument("--punct-count", dest="punct", action='store_true', help="Variety of words in output layer")
    parser.add_argument('--cuda', type=bool, default=False, help='cuda')    
    args = parser.parse_args()

    main(args)
