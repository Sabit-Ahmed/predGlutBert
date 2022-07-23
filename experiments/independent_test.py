import os

import pandas as pd
from proteinbert import OutputSpec, OutputType, InputEncoder

from experiments import utils
from experiments.model_utils import evaluate_by_len

out_spec = OutputSpec(OutputType(False, 'binary'), [0, 1])


def independent_test(ind_test_set_path, dataset_path, saved_model_path, ind_test_result_path):
    if not os.path.exists(ind_test_set_path):
        glut_data = utils.read_file(file_path=dataset_path, file_type='xlsx')
        train_set_primary, test_set = utils.split_data(data=glut_data, shuffle=True, test_size=0.1, random_state=42)
        test_set.to_csv(ind_test_set_path)
    else:
        test_set = pd.read_csv(ind_test_set_path)
        print('independent set reading done!')

    model_generator = utils.load_sav_file(file_path=saved_model_path)
    n_annotations = model_generator.pretraining_model_generator.n_annotations
    input_encoder = InputEncoder(n_annotations)
    results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, out_spec, test_set['seq'],
                                                test_set['label'],
                                                start_seq_len=512, start_batch_size=32)
    print(results)
    utils.get_performance(ind_test_result_path=ind_test_result_path, results=results, confusion_matrix=confusion_matrix)
