import os
import joblib
import pandas as pd
import numpy as np


# local files
import config

TEST_DATA = config.test_data
MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df["id"].values
    predictions = None



    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        label_encoders = joblib.load(os.path.join("../models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        for c in df.columns:
            lbl = label_encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
             
            label_encoders.append((c, lbl))

        clf = joblib.load(os.path.join("../models", f"{MODEL}_{FOLD}_.pkl"))
        cols = joblib.load(os.path.join("../models", f"{MODEL}_{FOLD}_columns.pkl"))

        preds = clf.predict_proba(df)[:, 1]
        # print(preds)
        if FOLD==0:
            predictions =  preds
        else:
            predictions += preds
    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns=["id", "target"])


if __name__=="__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index=False)
