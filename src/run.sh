export MODEL=$1

FOLD=0 python train.py
FOLD=1 python train.py
FOLD=2 python train.py
FOLD=3 python train.py
FOLD=4 python train.py