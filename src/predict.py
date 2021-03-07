import os
import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics

# local files
import config
import model_dispatcher


FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")

if __name__=="__main__":
    df = pd.read_csv(config.train_folds)
    train_df = df[df.kfold.isin(config.FOLD_MAPPING.get(FOLD))]
    valid_df = df[df.kfold==FOLD]

    ytrain = train_df.target.values
    yvalid = valid_df.target.values


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # specify this in config file (columns to drop)

    train_df = train_df.drop(["id", "target", "kfold"], axis=1)
    valid_df = valid_df.drop(["id", "target", "kfold"], axis=1)


    valid_df = valid_df[train_df.columns]

    label_encoders = []

    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
        train_df.loc[:, c] = lbl.transform(train_df[c].values.tolist())
        valid_df.loc[:, c] = lbl.transform(valid_df[c].values.tolist())   
        label_encoders.append((c, lbl))

    # data is ready to train
    clf =  model_dispatcher.MODELS[MODEL]
    clf.fit(train_df, ytrain)
    preds = clf.predict_proba(valid_df)[:, 1]
    print(metrics.roc_auc_score(yvalid, preds))


    joblib.dump(label_encoders, f"../models/{MODEL}_label_encoder.pkl")
    joblib.dump(clf, f"../models/{MODEL}.pkl")
