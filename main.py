from experiments.independent_test import independent_test
from experiments.training import training

# current_dir = os.getcwd()
customized_model_path = "models\customized_model\GlutBertModel.sav"
proteinBert_model_path = "models\proteinBert_model\epoch_92400_sample_23500000.pkl"
dataset_path = "data\iMul-kSite\multiLabelDataset.xlsx"
ind_test_set_path = "data\iMul-kSite\independent_set.csv"
ind_test_result_path = "results\independent_test.csv"

option = {1: 'training', 2: 'independent test', 3: 'both'}
print(f"1:{option[1]}\n2:{option[2]}\n3:{option[3]}\nWhat do you want to perform?", end="")
user_input = int(input())

if user_input == 1:
    training(dataset_path=dataset_path, saved_model_file_path=customized_model_path)

elif user_input == 2:
    independent_test(ind_test_set_path=ind_test_set_path, dataset_path=dataset_path,
                     saved_model_path=customized_model_path, ind_test_result_path=ind_test_result_path)
else:
    training(dataset_path=dataset_path, saved_model_file_path=customized_model_path)
    independent_test(ind_test_set_path=ind_test_set_path, dataset_path=dataset_path,
                     saved_model_path=customized_model_path, ind_test_result_path=ind_test_result_path)
