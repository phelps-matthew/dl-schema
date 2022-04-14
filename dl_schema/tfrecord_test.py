"""Write MNIST data to tfrecords"""
import argparse
from typing import Union
from pathlib import Path
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def write(
    labels: pd.DataFrame, img_root: Union[str, Path], output_path: Union[str, Path]
):
    """write MNIST tfrecords

    Args:
        labels: dataframe of MNIST labels
        img_root: root directory of MNIST images, referenced in labels
        output_path: tfrecord output file
    """
    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for i in tqdm(range(len(labels))):
            digit_id = labels["digit_id"][i]
            filename = labels["filename"][i]
            img_path = img_root / filename
            # load image as np.uint8, shape (28, 28)
            x = Image.open(img_path)
            x = np.array(x)
            height, width = x.shape
            # write tfexample
            tfexample = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "height": _int64_feature(height),
                        "width": _int64_feature(width),
                        "depth": _int64_feature(1),
                        "digit_id": _int64_feature(int(digit_id)),
                        "image_raw": _bytes_feature(x.tobytes()),
                    }
                )
            )
            writer.write(tfexample.SerializeToString())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Write MNIST tfrecords. To be used after extracting dataset via ./data/create_mnist_dataset.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./processed",
        help="root dir of processed MNIST data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tfrecords",
        help="target directory of output tfrecord files",
    )
    args = parser.parse_args()

    # set directories
    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    train_output_path = output_dir / "train.tfrecords"
    test_output_path = output_dir / "test.tfrecords"

    # load csv labels as dataframe
    train_labels = pd.read_csv(
        root_dir / "train/labels/annot.csv", header=None, names=["filename", "digit_id"]
    )
    test_labels = pd.read_csv(
        root_dir / "test/labels/annot.csv", header=None, names=["filename", "digit_id"]
    )

    # write tfrecords
    print(f"Writing MNIST train tfrecords to {train_output_path}")
    write(train_labels, root_dir / "train/images", train_output_path)
    print(f"Writing MNIST test tfrecords to {test_output_path}")
    write(test_labels, root_dir / "test/images", test_output_path)
