from numpy.core.numeric import full
from sklearn import preprocessing

"""
- label encoding
- one hot encoding
- binary encoding
"""

class CategoricalFeatures:
    def __init__(self, df, categorical_features, encoding_type, handle_na=False):
        """
        df: pandas Dataframe
        categorical_features: list of column names, e.g. ["ord1, "nom_0"....]
        encoding_type: label, binary, ohe
        handle_na: True/False
        """
        self.df = df
        self.cat_feats = categorical_features
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        # add a categorical selection function (user will input the whole dataframe)
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)
    
    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder(handle_unknown="ignore") # this might need to be changed in future
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df

    def _label_binarization(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelBinarizer()
            lbl.fit(self.df[c].values)
            val = lbl.transform(self.df[c].values)
            self.output_df = self.output_df.drop(c, axis=1) # output df
            for j in range(val.shape[1]):
                new_col_name = c + f"__bin_{j}"
                self.output_df[new_col_name] = val[:, j]
            self.binary_encoders[c] = lbl
        return self.output_df
    
    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder()
        ohe.fit(self.df[self.cat_feats].values)
        return ohe.transform(self.df[self.cat_feats].values)
    
    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        else:
            raise Exception("Encoding type not understood")
    
    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-999999")
        
        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        if self.enc_type == "binary":
            for c, lbl in self.label_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)
                for j in range(val.shape[1]):
                    new_col_name = c+ f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("../input/cat-in-the-dat/train.csv").head(500)
    df_test = pd.read_csv("../input/cat-in-the-dat/test.csv")
    train_len = len(df)
    df_test["target"] = -1
    full_data = pd.concat([df, df_test])
    cols = [c for c in df.columns if c not in ["id", "target"]]
    print(cols)
    cat_feats = CategoricalFeatures(df, 
                                    categorical_features=cols,
                                    encoding_type="ohe",
                                    handle_na=True)
    full_data_transformed = cat_feats.fit_transform()
    train_df = full_data_transformed[:train_len, :]
    test_df = full_data_transformed[train_len:, :]
    # print(output_df.columns)
    # print(output_df.bin_1.values)
    # print(output_df.bin_1.value_counts())

    # print(full_data_transformed.head())
    # print(train_df.head())
    # print(test_df.head())
    # print(len(full_data_transformed.columns))
    print(type(full_data_transformed))
    print(full_data_transformed[:10])