import math
import pickle

import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split


def read_file(file_path, file_type):
    if file_type == "xlsx":
        dataset = pd.read_excel(file_path, header=[8]).dropna().drop_duplicates()
        glut_data = pd.concat([dataset["Sequence"], dataset['Glut']], axis=1)
        glut_data.columns = ['seq', 'label']
        return glut_data

    if file_type == "txt":
        return pd.read_table(file_path, delimiter=" ")

    if file_type == "csv":
        return pd.read_csv(file_path)


def load_sav_file(file_path):
    model = pickle.load(open(file_path, 'rb'))
    return model


def split_data(data=None, test_size=0.1, random_state=0, shuffle=True, stratify=None):
    train_set, valid_set = train_test_split(data, shuffle=shuffle, test_size=test_size, random_state=random_state,
                                            stratify=stratify)
    return train_set, valid_set


def get_performance(ind_test_result_path, results, confusion_matrix):
    print('Test-set performance:')
    display(results)

    print('Confusion matrix:')
    display(confusion_matrix)

    TP = confusion_matrix.iloc[1, 1]
    TN = confusion_matrix.iloc[0, 0]
    FP = confusion_matrix.iloc[0, 1]
    FN = confusion_matrix.iloc[1, 0]

    SP = TN / (TN + FP)
    SN = Recall = TP / (TP + FN)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    Precision = TP / (TP + FP)
    MCC = ((TP * TN) - (FP * FN)) / math.sqrt((TP + FN) * (TN + FN) * (TP + FP) * (TN + FP))
    F1_Score = (2 * Precision * Recall) / (Precision + Recall)

    result_df = pd.DataFrame([SP, SN, ACC, MCC, F1_Score])
    result_df = result_df.T
    result_df.columns = ['SP', 'SN', 'ACC', 'MCC', 'F1-Score']
    # if not os.path.isdir(results_dir):
    #     os.mkdir(results_dir)
    result_df.to_csv(ind_test_result_path)
