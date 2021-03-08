import os
import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics

# local files
import config
import model_dispatcher

TRAIN_DATA = config.train_folds
TEST_DATA = config.test_data
MODEL = os.environ.get("MODEL")
FOLD = int(os.environ.get("FOLD"))

if __name__=="__main__":
    df = pd.read_csv(TEST_DATA)



    for FOLD in range(5):
        label_encoders = joblib.load(os.path.join("models", f"../models/{MODEL}_{FOLD}_label_encoder.pkl"))
        for c in df.columns:
            lbl = label_encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
             
            label_encoders.append((c, lbl))

        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))

        preds = clf.predict_proba(df)[:, 1]
        print(preds)


