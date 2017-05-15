import torch

from be_gan_trainer import Trainer
from be_gan_config import get_config
from be_gan_data_loader import get_loader
from be_gan_utils import prepare_dirs_and_logger, save_config



def main(config):
    torch.cuda.set_device(0)
    prepare_dirs_and_logger(config)

    torch.manual_seed(config.random_seed)
    if config.num_gpu > 0:
        torch.cuda.manual_seed(config.random_seed)

    if config.is_train:
        data_path = config.data_path
        batch_size = config.batch_size
        do_shuffle = True
    else:
        if config.test_data_path is None:
            data_path = config.data_path
        else:
            data_path = config.test_data_path
        batch_size = config.sample_per_image
        do_shuffle = False

    data_loader = get_loader(
        data_path, config.split, batch_size, config.input_scale_size, config.num_worker, do_shuffle)

    trainer = Trainer(config, data_loader)

    if config.is_train:
        save_config(config)
        trainer.train()
    else:
        if not config.load_path:
            raise Exception("[!] You should specify `load_path` to load a pretrained model")
        trainer.test()

if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
