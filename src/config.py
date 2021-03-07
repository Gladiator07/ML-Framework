train_csv = "../input/train.csv"   # train.csv file location
n_fold_splits = 5                  # number of folds 
train_folds="../input/train_folds.csv"  # train folds file location
test_data = "../input/test.csv"
FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3]
}                             # mapping for folds (if you have 10 folds you can change it here)