import shutil
import os
import json
import tensorflow as tf
from submarine.pipeline.model import deepFM
from submarine.pipeline.input import libsvm_input_fn

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("model_dir", '', "model check point dir")


def main(_):
    data_dir = './data/'
    train_data = data_dir + 'tr.libsvm'
    valid_data = data_dir + 'va.libsvm'
    test_data = data_dir + 'te.libsvm'
    json_path = './deepFM.json'

    model_dir = FLAGS.model_dir
    if tf.gfile.Exists(model_dir):
        tf.gfile.DeleteRecursively(model_dir)
    tf.gfile.MakeDirs(model_dir)

    parameters = {
        "field_size": 39,
        "embedding_size": 256,
        "learning_rate": 0.0005,
    }

    model = deepFM(model_dir=model_dir, model_params=parameters, json_path=json_path)
    # Training
    model.train(libsvm_input_fn(train_data), libsvm_input_fn(valid_data))
    # Evaluate
    model.evaluate(libsvm_input_fn(valid_data))
    # Predict
    model.predict(libsvm_input_fn(test_data), predict_keys="prob")


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
