python3 train.py -tr ../data/fold_0/train.tsv --emb ../En_vectors.txt -tu ../data/fold_0/dev.tsv -ts ../data/fold_0/test.tsv -p 1 -o output_dir --cuda -b 40 -t bregp --epochs 100 --compressed_datasets ../datasets-pickled.pkl --nm newpa
# ensembles
python3 train.py -tr ../data/fold_0/train.tsv -tu ../data/fold_0/dev.tsv -ts ../data/fold_0/test.tsv -o out_ensemble1/ -p 1 --epochs 5 --pos --variety --punct-count --ensembles run.cnn.0/models/modelbgrepproper.19.pt run.cnn.0.pos/models/modelbgrepproper.48.pt run.cnn.0.pos/models/modelbgrepproper.8.pt run.cnn.0.punct/models/modelbgrepproper.31.pt run.cnn.0.variety/models/modelbgrepproper.23.pt

Input QWKs:
run.cnn.0/models/modelbgrepproper.19.pt : 0.42
run.cnn.0.pos/models/modelbgrepproper.48.pt : 0.43
run.cnn.0.pos/models/modelbgrepproper.8.pt : 0.46
run.cnn.0.punct/models/modelbgrepproper.31.pt : 0.58
run.cnn.0.variety/models/modelbgrepproper.23.pt : 0.42
Ensemble QWKs:
Rank : 0 Score: 0.543631 : Name: modelbgrepproper.6.pt
Rank : 1 Score: 0.528709 : Name: modelbgrepproper.7.pt
Rank : 2 Score: 0.523145 : Name: modelbgrepproper.pt
Rank : 3 Score: 0.509986 : Name: modelbgrepproper.4.pt
Rank : 4 Score: 0.483777 : Name: modelbgrepproper.9.pt
Rank : 5 Score: 0.482674 : Name: modelbgrepproper.1.pt
Rank : 6 Score: 0.455729 : Name: modelbgrepproper.2.pt
Rank : 7 Score: 0.446038 : Name: modelbgrepproper.3.pt
Rank : 8 Score: 0.421331 : Name: modelbgrepproper.0.pt
Rank : 9 Score: 0.410119 : Name: modelbgrepproper.5.pt

# Supervisor 1
python3 train.py -tr ../data/fold_0/train.tsv -tu ../data/fold_0/dev.tsv -ts ../data/fold_0/test.tsv -o out_supervisor/ -p 1 --epochs 5 --pos --variety --punct-count --ensembles run.cnn.0.pos.variety/models/modelbgrepproper.48.pt run.cnn.0.punct/models/modelbgrepproper.31.pt run.cnn.100.pos.variety.punct/models/modelbgrepproper.76.pt --ensemble-method supervisor --cuda
Input nets:
run.cnn.0.pos.variety/models/modelbgrepproper.48.pt 0.46
run.cnn.0.punct/models/modelbgrepproper.31.pt 0.58
run.cnn.100.pos.variety.punct/models/modelbgrepproper.76.pt 0.62
Outputs:
Rank : 0 Score: 0.565078 : Name: modelbgrepproper.3.pt
Rank : 1 Score: 0.541309 : Name: modelbgrepproper.1.pt
Rank : 2 Score: 0.486148 : Name: modelbgrepproper.4.pt
Rank : 3 Score: 0.476125 : Name: modelbgrepproper.pt
Rank : 4 Score: 0.463662 : Name: modelbgrepproper.0.pt
Rank : 5 Score: 0.427911 : Name: modelbgrepproper.2.pt


# Supervisor 2
python3 train.py -tr ../data/fold_0/train.tsv -tu ../data/fold_0/dev.tsv -ts ../data/fold_0/test.tsv -o out_supervisor2/ -p 1 --epochs 5 --pos --variety --punct-count --ensembles run.cnn.100.variety.punct/models/modelbgrepproper.14.pt run.cnn.0.punct/models/modelbgrepproper.31.pt run.cnn.100.pos.variety.punct/models/modelbgrepproper.76.pt --ensemble-method supervisor --cuda
Input nets:
run.cnn.100.variety.punct/models/modelbgrepproper.14.pt 0.56
run.cnn.0.punct/models/modelbgrepproper.31.pt 0.58
run.cnn.100.pos.variety.punct/models/modelbgrepproper.76.pt 0.62

python3 qwkscorer.py -m ./out_ensemble1/models -r ../data/fold_0/train.tsv -t ../data/fold_0/test.tsv -d ../data/fold_0/dev.tsv --prompt 1 -v 4000 -b 128 --pos --variety --punct-count --cuda False
