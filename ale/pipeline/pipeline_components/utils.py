from ale.config import AppConfig


def create_path(data_dir, train_file, file_format):
    return f"{data_dir}/{train_file}.{file_format}"


def prepare_data(cfg: AppConfig):
    train_path = create_path(
        cfg.data.data_dir, cfg.data.train_file, cfg.data.file_format
    )
    dev_path = create_path(cfg.data.data_dir, cfg.data.dev_file, cfg.data.file_format)
    test_path = create_path(cfg.data.data_dir, cfg.data.test_file, cfg.data.file_format)
    return train_path, dev_path, test_path
