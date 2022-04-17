"""sample tensorflow dataset from tfrecords"""
import tensorflow as tf
from pathlib import Path

def parse_fn(example_proto):
    """parse single example proto from tfrecords"""
    features = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
        "digit_id": tf.io.FixedLenFeature([], tf.int64),
        "image_raw": tf.io.FixedLenFeature([], tf.string),
    }
    features_parsed = tf.io.parse_single_example(example_proto, features)
    # decode ints
    label = tf.cast(features_parsed["digit_id"], tf.int32)
    width = tf.cast(features_parsed["width"], tf.int32)
    height = tf.cast(features_parsed["height"], tf.int32)
    depth = tf.cast(features_parsed["depth"], tf.int32)
    # decode image
    image = tf.io.decode_raw(features_parsed["image_raw"], tf.uint8)
    image = tf.reshape(image, [height, width, depth])
    image = tf.cast(image, tf.float32)
    # normalize to zero mean and unit std
    image = tf.image.per_image_standardization(image)

    return image, label


def mnist_dataset(cfg, train=True):
    """generate tf mnist dataset based on cfg"""
    # get tfrecord path
    tfr_path = cfg.data.train_tfrecords_path if train else cfg.data.test_tfrecords_path
    # if relative, make relative to dl_schema dir
    if Path(tfr_path).is_absolute():
        tfr_path = Path(tfr_path)
    else:
        dl_schema_dir = Path(__file__).parent
        tfr_path = Path(dl_schema_dir / tfr_path).expanduser().resolve()
    # create tf dataset and decode example protos; shuffle and set batch size
    if cfg.data.shuffle and train:
        dataset = tf.data.TFRecordDataset(tfr_path).map(parse_fn)
        dataset = dataset.shuffle(buffer_size=cfg.data.buffer_size).batch(cfg.bs)
    else:
        dataset = tf.data.TFRecordDataset(tfr_path).map(parse_fn).batch(cfg.bs)
    return dataset
