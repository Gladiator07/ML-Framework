from numpy import mod
import pandas as pd
from sklearn import model_selection

# local libraries
import config

if __name__=="__main__":
    df = pd.read_csv(config.train_csv)
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    kf = model_selection.StratifiedKFold(n_splits=config.n_fold_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_dix) in enumerate(kf.split(X=df, y=df.target.values)):
        print(len(train_idx), len(val_dix))
        df.loc[val_dix, 'kfold'] = fold
    
    df.to_csv("../input/train_folds.csv", index=False)
     