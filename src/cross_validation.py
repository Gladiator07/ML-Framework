class CrossValidation:
    def __init__(self, df, target_cols):
        self.dataframe = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)

    def split(self):
        if len(self.num_targets) == 1:
            unique_values = self.dataframe[self.target_cols[0]].nunique()
            if unique_values == 1:
                raise Exception("Only one unique value found!")
            #### TO BE CONTINUED SOON .......
            #### FIRST FINISHING THE CATEGORICAL ENCODING PART