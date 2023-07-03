import datetime


class ProjectConfig:
    def __init__(self) -> None:

        self.config_yaml = None

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tfboard_train_log_dir = "log/{}/train".format(current_time)
        self.tfboard_test_log_dir = "log/{}/test".format(current_time)
        self.tfboard_log_dir = "log/{}".format(current_time)
