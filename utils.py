import os

import pandas as pd


def read_file(directory, file_type, file_name):
    current_dir = os.getcwd()
    dataset_dir = current_dir + f"\{directory}\{file_name}.{file_type}"

    if file_type == "xlsx":
        return pd.read_excel(dataset_dir, sheet_name=None)

    if file_type == "txt":
        return pd.read_table(dataset_dir, delimiter=" ")

    if file_type == "csv":
        return pd.read_csv(dataset_dir)
