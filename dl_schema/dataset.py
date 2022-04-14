"""sample tensorflow dataset from tfrecords"""
import tensorflow as tf

def parse_fn(example_proto):
    # Create a dictionary describing the features.
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'digit_id': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
    }
    # Parse the input tf.train.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_feature_description)

def mnist_dataset(path):
    tfrecords_path = cfg.train_tfrecords_path if train else cfg.test_tfrecords_path
    dataset = tf.data.TFRecordDataset(tfrecords_path).map(parse_fn)
    return dataset
