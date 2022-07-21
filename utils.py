import os

import pandas as pd
from sklearn.model_selection import train_test_split


def read_file(directory, file_type, file_name):
    current_dir = os.getcwd()
    data_dir = current_dir + f"\{directory}\{file_name}.{file_type}"

    if file_type == "xlsx":
        return pd.read_excel(data_dir, header=[8]).dropna().drop_duplicates()

    if file_type == "txt":
        return pd.read_table(data_dir, delimiter=" ")

    if file_type == "csv":
        return pd.read_csv(data_dir)


def split_data(data=None, test_size=0.1, random_state=0, shuffle=True, stratify=None):
    train_set, valid_set = train_test_split(data, shuffle=shuffle, test_size=test_size, random_state=random_state,
                                            stratify=stratify)
    return train_set, valid_set
