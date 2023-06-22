from src.config import ProjectConfig

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D


class BaseModel(tf.keras.Model):
    def __init__(self, project_conf: ProjectConfig) -> None:
        super(BaseModel, self).__init__()
        self.project_conf = project_conf

        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.dense_1 = Dense(128, activation='relu')
        self.dense_2 = Dense(self.project_conf.num_classes)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.flatten(x)
        x = self.dense_1(x)
        return self.dense_2(x)
