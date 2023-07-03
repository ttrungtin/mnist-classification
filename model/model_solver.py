from model.model_config import *
from data.data_solver import *

import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy, Mean
from tensorflow.keras.callbacks import TensorBoard


class ModelSolver:

    def __init__(self, project_conf) -> None:
        self.project_conf = project_conf
        self.model = self.load_model()

    def load_model(self):
        model_name = self.project_conf.config_yaml["MODEL"]["NAME"]
        if model_name == "base":
            return BaseModel(self.project_conf)
        if model_name == "cnn":
            return CNNModel(self.project_conf)
        pass

    def do_train(self, data_solver: RawDataProcessor) -> None:
        data_solver.load_data_tfds()

        self.model.compile(
            optimizer=Adam(),
            loss=SparseCategoricalCrossentropy(from_logits=True),
            metrics=[SparseCategoricalAccuracy()]
        )

        # callback
        tfboard = TensorBoard(
            log_dir=self.project_conf.tfboard_log_dir, histogram_freq=1)

        # fit
        self.model.fit(
            data_solver.ds_train,
            epochs=self.project_conf.config_yaml["TRAIN"]["EPOCHS"],
            validation_data=data_solver.ds_test,
            callbacks=[tfboard]
        )
