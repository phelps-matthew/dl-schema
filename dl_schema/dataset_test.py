from dl_schema.utils import parse_fn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

train_dataset = tf.data.TFRecordDataset("./data/tfrecords/train.tfrecords").map(parse_fn)
for data in train_dataset.take(2):
    print(repr(data))
    # fmt: off
    import ipdb; ipdb.set_trace(context=30)  # noqa
    # fmt: on
