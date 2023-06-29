import datetime


class ProjectConfig:
    batch_size = 64
    num_classes = 10
    epochs = 100

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tfboard_train_log_dir = "log/{}/train".format(current_time)
    tfboard_test_log_dir = "log/{}/test".format(current_time)
    tfboard_log_dir = "log/{}".format(current_time)
