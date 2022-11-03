import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


class version(object):
    def __init__(self, data_path="data/corpus"):
        self.data_path = data_path

    def load(self, url):
        # Get
        raw = pd.read_csv(url)
        # Vote
        raw["label"] = (
            raw.hatespeech_G1 + raw.hatespeech_G2 + raw.hatespeech_G3
        ).apply(lambda x: 1 if x >= 2 else 0)
        # Save
        df_raw[["text", "label"]].to_csv(f"{self.data_path}/raw_data.csv", index=False)

    def split(self, dataframe=False, test_size=0.1):

        # Data version path
        dvc_path = f"{self.data_path}/test_size_{str(int(test_size*100))}"

        # If dataframe is in dataframe
        if dataframe:
            return "dataframe splited"

        # First split
        df_raw = pd.read_csv(f"{self.data_path}/raw_data.csv")
        if not os.path.exists(dvc_path):
            # Train and test
            df_train, df_test = train_test_split(
                df_raw, stratify=df_raw["label"], test_size=test_size, random_state=42
            )
            # Save in the correct path
            Path(f"{dvc_path}").mkdir(parents=True, exist_ok=True)
            df_train.to_csv(f"{dvc_path}/train_data.csv", index=False)
            df_test.to_csv(f"{dvc_path}/test_data.csv", index=False)

        # If data alredy splited
        else:
            df_train = pd.read_csv(f"{dvc_path}/train_data.csv")
            df_test = pd.read_csv(f"{dvc_path}/test_data.csv")

        return (df_train, df_test)
