python train.py -tr data/fold_0/train.tsv -tu data/fold_0/dev.tsv -ts data/fold_0/test.tsv -p 1 -o output_dir -b 8 -t bregp

# Debugging
python train.py -tr ../data/fold_0/train.tsv -tu ../data/fold_0/dev.tsv -ts ../data/fold_0/test.tsv -p 1 -o output_dir -b 8 -t bregp --pos --variety --punct-count -v 4000 --compressed_datasets 'datasets-pickled.pkl'
