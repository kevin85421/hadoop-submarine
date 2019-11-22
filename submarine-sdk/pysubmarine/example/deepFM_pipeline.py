import shutil
from os import path
import tensorflow as tf
from submarine.pipeline.model import deepFM


def make_input_fn(filenames, batch_size=256, num_epochs=2, perform_shuffle=False):
    def _input_fn():
        def decode_libsvm(line):
            columns = tf.string_split([line], ' ')
            labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
            splits = tf.string_split(columns.values[1:], ':')
            id_vals = tf.reshape(splits.values, splits.dense_shape)
            feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
            feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
            feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)
            return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

        # Extract lines from input files using the Dataset API, can pass one filename or filename list
        dataset = tf.data.TextLineDataset(filenames).map(decode_libsvm, num_parallel_calls=10).prefetch(500000)

        # Randomizes input using a window of 256 elements (read into memory)
        if perform_shuffle:
            dataset = dataset.shuffle(buffer_size=256)

        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        batch_features, batch_labels = iterator.get_next()
        return batch_features, batch_labels
    return _input_fn


def main(_):
    data_dir = './data/'
    train_data = data_dir + 'tr.libsvm'
    valid_data = data_dir + 'va.libsvm'
    test_data = data_dir + 'te.libsvm'

    model_dir = './DeepFM'
    if path.exists(model_dir):
        shutil.rmtree(model_dir)
    model = deepFM(model_dir=model_dir)

    # Training
    model.train(make_input_fn(train_data), make_input_fn(valid_data))
    # Evaluate
    model.evaluate(make_input_fn(valid_data))
    # Predict
    model.predict(make_input_fn(test_data))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
