import argparse
import datetime

from model.model_base import BaseModel
from src.raw_data_processing import RawDataProcessor
from src.config import ProjectConfig

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean


class ModelTrainer:

    def __init__(self, project_conf: ProjectConfig) -> None:
        self.project_conf = project_conf
        self.ds_processor = None

    def load_data(self):
        self.ds_processor = RawDataProcessor(self.project_conf)
        self.ds_processor.load_data_tfmn()

    def train_model(self):
        # load data
        self.load_data()

        # load model
        model = BaseModel(self.project_conf)

        # config
        loss = SparseCategoricalCrossentropy(from_logits=True)
        optimizer = Adam()

        train_loss = Mean(name='train_loss')
        train_acc = SparseCategoricalAccuracy(name='train_acc')

        test_loss = Mean(name='test_loss')
        test_acc = SparseCategoricalAccuracy(name='test_acc')

        # tensorboard
        train_summary_writer = tf.summary.create_file_writer(
            self.project_conf.tfboard_train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(
            self.project_conf.tfboard_test_log_dir)

        # step
        EPOCHS = self.project_conf.epochs
        for epochs in range(EPOCHS):
            train_loss.reset_states()
            train_acc.reset_states()
            test_loss.reset_states()
            test_acc.reset_states()

            # train
            for images, labels in self.ds_processor.ds_train:
                with tf.GradientTape() as tape:
                    predictions = model(images, training=True)
                    losses = loss(labels, predictions)
                gradients = tape.gradient(losses, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(gradients, model.trainable_variables))

                # for print
                train_loss(losses)
                train_acc(labels, predictions)

            # test
            for test_images, test_labels in self.ds_processor.ds_test:
                predictions = model(test_images, training=False)
                losses = loss(test_labels, predictions)

                # for print
                test_loss(losses)
                test_acc(test_labels, predictions)

            # tensorboard
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epochs)
                tf.summary.scalar('acc', train_acc.result(), step=epochs)
            with test_summary_writer.as_default():
                tf.summary.scalar("test loss", test_loss.result(), step=epochs)
                tf.summary.scalar("test acc", test_acc.result(), step=epochs)

            print("Epoch: {} Loss: {} Acc: {} Test Loss: {} Test Acc: {}".format(
                epochs+1,
                train_loss.result(),
                train_acc.result() * 100,
                test_loss.result(),
                test_acc.result() * 100
            )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch-size", type=int,
                        default=ProjectConfig.batch_size)
    parser.add_argument("-e", "--epochs", type=int,
                        default=ProjectConfig.epochs)
    parser.add_argument("-c", "--num-classes", type=int,
                        default=ProjectConfig.num_classes)
    args = parser.parse_args()

    project_conf = ProjectConfig()
    project_conf.batch_size = args.batch_size
    project_conf.epochs = args.epochs
    project_conf.num_classes = args.num_classes

    model_trainer = ModelTrainer(project_conf)
    model_trainer.train_model()
