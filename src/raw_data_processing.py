import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

from src.config import ProjectConfig


class RawDataProcessor:
    def __init__(self, project_conf: ProjectConfig) -> None:
        self.ds_train = None
        self.ds_test = None
        self.ds_info = None
        self.project_conf = project_conf

    def load_data_tfds(self):
        (self.ds_train, self.ds_test), self.ds_info = tfds.load(
            'mnist',
            split=['train', 'test'],
            shuffle_files=True,
            as_supervised=True,
            with_info=True
        )

        # train gen
        self.ds_train = self.ds_train.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_train = self.ds_train.cache()
        self.ds_train = self.ds_train.shuffle(
            self.ds_info.splits['train'].num_examples)
        self.ds_train = self.ds_train.batch(self.project_conf.batch_size)
        self.ds_train = self.ds_train.prefetch(tf.data.AUTOTUNE)

        # test gen
        self.ds_test = self.ds_test.map(
            self.normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
        self.ds_test = self.ds_test.batch(self.project_conf.batch_size)
        self.ds_test = self.ds_test.cache()
        self.ds_test = self.ds_test.prefetch(tf.data.AUTOTUNE)

    def load_data_tfmn(self):
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        x_train = x_train[..., tf.newaxis].astype("float32")
        x_test = x_test[..., tf.newaxis].astype("float32")

        # train gen
        self.ds_train = tf.data.Dataset.from_tensor_slices(
            (x_train, y_train)).shuffle(10000).batch(self.project_conf.batch_size)

        # test gen
        self.ds_test = tf.data.Dataset.from_tensor_slices(
            (x_test, y_test)).batch(self.project_conf.batch_size)

    @staticmethod
    def normalize_img(image, label):
        return tf.cast(image, tf.float32) / 255., label

    def print_data_info(self):
        print(self.ds_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int,
                        default=ProjectConfig.batch_size)
    args = parser.parse_args()

    project_conf = ProjectConfig()
    project_conf.batch_size = args.batch_size

    ds_processor = RawDataProcessor(project_conf)
    ds_processor.load_data_tfds()
    ds_processor.print_data_info()
