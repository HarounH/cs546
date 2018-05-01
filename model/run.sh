python3 train.py -tr ../data/fold_0/train.tsv --emb ../En_vectors.txt -tu ../data/fold_0/dev.tsv -ts ../data/fold_0/test.tsv -p 1 -o output_dir --cuda -b 40 -t bregp --epochs 100 --compressed_datasets ../datasets-pickled.pkl --nm newpa
# ensembles
python3 train.py -tr ../data/fold_0/train.tsv -tu ../data/fold_0/dev.tsv -ts ../data/fold_0/test.tsv -o out/ -p 1 -e 2 --pos --variety --punct-count\
--ensembles run.cnn.0/models/modelbgrepproper.19.pt run.cnn.0.pos/models/modelbgrepproper.48.pt run.cnn.0.pos/models/modelbgrepproper.8.pt run.cnn.0.punct/models/modelbgrepproper.31.pt run.cnn.0.variety/models/modelbgrepproper.23.pt
