import os

import numpy as np
import pandas as pd
from proteinbert import conv_and_global_attention_model, load_pretrained_model_from_dump, load_pretrained_model
from proteinbert.finetuning import encode_dataset, get_evaluation_results, split_dataset_by_len
from tensorflow import keras

current_dir = os.getcwd()


def load_proteinbert_model():
    model_dir = current_dir + "\proteinbert_models"
    pretrained_model_file_path = model_dir + "\epoch_92400_sample_23500000.pkl"

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    if not os.path.exists(pretrained_model_file_path):
        pretrained_model_generator, input_encoder = load_pretrained_model(
            local_model_dump_dir=model_dir,
            local_model_dump_file_name='epoch_92400_sample_23500000.pkl')
    else:
        pretrained_model_generator, input_encoder = load_pretrained_model_from_dump(
            dump_file_path=pretrained_model_file_path,
            create_model_function=conv_and_global_attention_model.create_model,
            create_model_kwargs={},
            optimizer_class=keras.optimizers.Adam, lr=2e-04,
            other_optimizer_kwargs={},
            annots_loss_weight=1,
            load_optimizer_weights=False)
    return pretrained_model_generator, input_encoder


def evaluate_by_len(model_generator, input_encoder, output_spec, seqs, raw_Y, start_seq_len=512, start_batch_size=32,
                    increase_factor=2):
    assert model_generator.optimizer_weights is None

    dataset = pd.DataFrame({'seq': seqs, 'raw_y': raw_Y})

    results = []
    results_names = []
    y_trues = []
    y_preds = []

    for len_matching_dataset, seq_len, batch_size in split_dataset_by_len(dataset, start_seq_len=start_seq_len,
                                                                          start_batch_size=start_batch_size, \
                                                                          increase_factor=increase_factor):

        X, y_true, sample_weights = encode_dataset(len_matching_dataset['seq'], len_matching_dataset['raw_y'],
                                                   input_encoder, output_spec, \
                                                   seq_len=seq_len, needs_filtering=False)

        assert set(np.unique(sample_weights)) <= {0.0, 1.0}
        y_mask = (sample_weights == 1)

        model = model_generator.create_model(seq_len)
        y_pred = model.predict(X, batch_size=batch_size)

        y_true = y_true[y_mask].flatten()
        y_pred = y_pred[y_mask]

        if output_spec.output_type.is_categorical:
            y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        else:
            y_pred = y_pred.flatten()

        results.append(get_evaluation_results(y_true, y_pred, output_spec))
        results_names.append(seq_len)

        y_trues.append(y_true)
        y_preds.append(y_pred)

    y_true = np.concatenate(y_trues, axis=0)
    y_pred = np.concatenate(y_preds, axis=0)
    all_results, confusion_matrix = get_evaluation_results(y_true, y_pred, output_spec, return_confusion_matrix=True)
    results.append(all_results)
    results_names.append('All')

    results = pd.DataFrame(results, index=results_names)
    results.index.name = 'Model seq len'

    return results, confusion_matrix
