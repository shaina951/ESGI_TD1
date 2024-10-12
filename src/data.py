import pandas as pd

def make_dataset(filename="names_train"):
    return pd.read_csv(filename)
