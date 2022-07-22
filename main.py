import math
import os
import pickle

import pandas as pd
from IPython.display import display
from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, \
    evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from tensorflow import keras

import model_utils
import utils

fileName = 'multiLabelDataset'
current_dir = os.getcwd()
dataset_dir = current_dir + '\data'
dataset_file_path = os.path.join(dataset_dir, '%s.xlsx' % fileName)
model_dir = current_dir + "\proteinbert_models"
pretrained_model_file_path = model_dir + '\epoch_92400_sample_23500000.pkl'
final_model_file_path = model_dir + '\GlutBertModel.sav'
results_dir = current_dir + r'\results'

dataset = utils.read_file('data', 'xlsx', fileName)

glut_data = pd.concat([dataset["Sequence"], dataset['Glut']], axis=1)
glut_data.columns = ['seq', 'label']

train_set_primary, test_set = utils.split_data(data=glut_data, shuffle=True, test_size=0.1, random_state=42)
train_set, valid_set = utils.split_data(data=train_set_primary, stratify=train_set_primary['label'], test_size=0.1,
                                        random_state=0)

# A local (non-global) bianry output
OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
if not os.path.exists(pretrained_model_file_path):
    pretrained_model_generator, input_encoder = load_pretrained_model(local_model_dump_dir=model_dir,
                                                                      local_model_dump_file_name='epoch_92400_sample_23500000.pkl')
else:
    pretrained_model_generator, input_encoder = model_utils.load_model_from_local(pretrained_model_file_path)

# get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                           pretraining_model_manipulation_function=
                                           get_model_with_hidden_layers_as_outputs, dropout_rate=0.5)

training_callbacks = [
    keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
    keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
]

finetune(model_generator, input_encoder, OUTPUT_SPEC, train_set['seq'], train_set['label'], valid_set['seq'],
         valid_set['label'], seq_len=512, batch_size=32, max_epochs_per_stage=10, lr=1e-04,
         begin_with_frozen_pretrained_layers=True,
         lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
         callbacks=training_callbacks)

pickle.dump(model_generator, open(final_model_file_path, 'wb'))

results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_set['seq'],
                                            test_set['label'],
                                            start_seq_len=512, start_batch_size=32)
# print('Test-set performance:')
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
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
result_df.to_csv(results_dir + '\independent_test.csv')
