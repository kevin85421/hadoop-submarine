import shutil
from os import path
from submarine.pipeline.model import deepFM
from submarine.pipeline.input import libsvm_input_fn


if __name__ == "__main__":
    data_dir = './data/'
    train_data = data_dir + 'tr.libsvm'
    valid_data = data_dir + 'va.libsvm'
    test_data = data_dir + 'te.libsvm'

    model_dir = './DeepFM'
    if path.exists(model_dir):
        shutil.rmtree(model_dir)

    Parameters = {
        "field_size": 38,
        "embedding_size": 256,
        "learning_rate": 0.0005,
    }
    model = deepFM(model_dir=model_dir, model_params=Parameters)

    # Training
    model.train(libsvm_input_fn(train_data), libsvm_input_fn(valid_data))
    # Evaluate
    model.evaluate(libsvm_input_fn(valid_data))
    # Predict
    model.predict(libsvm_input_fn(test_data))
