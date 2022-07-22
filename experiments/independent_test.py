import os

import pandas as pd
from proteinbert import OutputSpec, OutputType

from experiments import utils
from experiments.model_utils import evaluate_by_len

model_dir = "proteinbert_models"
out_spec = OutputSpec(OutputType(False, 'binary'), [0, 1])

data_dir = "data\iMul-kSite"
ind_test_name = "\ind_test_set"
ind_set_path = data_dir + ind_test_name
dataset_name = "\multiLabelDataset"
dataset_path = data_dir + dataset_name

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)
if not os.path.exists(ind_set_path):
    glut_data = utils.read_file(file_path=dataset_path, file_type='xlsx')
    train_set_primary, test_set = utils.split_data(data=glut_data, shuffle=True, test_size=0.1, random_state=42)
    test_set.to_csv(ind_set_path)
else:
    ind_set = pd.read_csv(ind_set_path)
    print('independent set reading done!')
    model_generator, input_encoder = utils.load_sav_file(file_path=model_dir)
    results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, out_spec, ind_set['seq'],
                                                ind_set['label'],
                                                start_seq_len=512, start_batch_size=32)
