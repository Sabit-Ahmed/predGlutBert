import pickle

from proteinbert import FinetuningModelGenerator, finetune, OutputSpec, OutputType
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from tensorflow import keras

from experiments import utils, model_utils

out_spec = OutputSpec(OutputType(False, 'binary'), [0, 1])


def training(dataset_path, saved_model_file_path):
    dataset = utils.read_file(dataset_path, 'xlsx')

    train_set_primary, test_set = utils.split_data(data=dataset, shuffle=True, test_size=0.1, random_state=42)
    train_set, valid_set = utils.split_data(data=train_set_primary, stratify=train_set_primary['label'], test_size=0.1,
                                            random_state=0)

    pretrained_model_generator, input_encoder = model_utils.load_proteinbert_model()

    model_generator = FinetuningModelGenerator(
        pretraining_model_generator=pretrained_model_generator,
        output_spec=out_spec,
        pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
        dropout_rate=0.5)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
    ]

    finetune(model_generator, input_encoder, out_spec, train_set['seq'], train_set['label'], valid_set['seq'],
             valid_set['label'], seq_len=512, batch_size=32, max_epochs_per_stage=10, lr=1e-04,
             begin_with_frozen_pretrained_layers=True,
             lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
             callbacks=training_callbacks)

    pickle.dump(model_generator, open(saved_model_file_path, 'wb'))
    print('model training is done!')
