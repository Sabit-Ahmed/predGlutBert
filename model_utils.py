from proteinbert import conv_and_global_attention_model, load_pretrained_model_from_dump
from tensorflow import keras


def load_model_from_local(model_file_path):
    create_model_function = conv_and_global_attention_model.create_model
    create_model_kwargs = {}
    optimizer_class = keras.optimizers.Adam
    lr = 2e-04
    other_optimizer_kwargs = {}
    annots_loss_weight = 1
    load_optimizer_weights = False

    model_generator, input_encoder = load_pretrained_model_from_dump(dump_file_path=model_file_path,
                                                                     create_model_function=create_model_function,
                                                                     create_model_kwargs=create_model_kwargs,
                                                                     optimizer_class=optimizer_class, lr=lr,
                                                                     other_optimizer_kwargs=other_optimizer_kwargs,
                                                                     annots_loss_weight=annots_loss_weight,
                                                                     load_optimizer_weights=load_optimizer_weights)
    return model_generator, input_encoder
