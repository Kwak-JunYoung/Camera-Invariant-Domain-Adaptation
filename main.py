from models.unet import UNet
from torch.utils.data import Dataset, DataLoader
from data_loaders import CustomDataset
import numpy as np
import wandb
import torch
import random
import time
from time import localtime
from utils.config import ConfigNode as CN
import yaml
import os
import pandas as pd
import argparse


# https://huggingface.co/docs/transformers/v4.32.0/ko/accelerate
from accelerate import Accelerator
from utils.file_io import PathManager
from torch.optim import Adam
from train import model_train

# Random seed


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

# Obtain information of the model


def get_model_info(config, device, model_name):
    if model_name == "unet":
        model_config = config.unet_config
        model = UNet("nothing")
    else:
        raise NotImplementedError("model name is not valid")
    return model_config, model


def get_data_info(accelerator, file_dir, shuffle, infer, num_workers, batch_size, device):
    dataset = CustomDataset(csv_file=file_dir, infer=infer)

    dataloader = accelerator.prepare(DataLoader(dataset=dataset, batch_size=batch_size,
                                                shuffle=shuffle, num_workers=num_workers, generator=torch.Generator(device=device)))
    return dataset, dataloader

# Get test results to record in wandb


def get_print_args(test_aucs, test_accs, config, train_config, model_config):
    model_name = config.model_name
    data_name = config.data_name

    test_auc = np.mean(test_aucs)
    test_auc_std = np.std(test_aucs)
    test_acc = np.mean(test_accs)
    test_acc_std = np.std(test_accs)

    print("\n5-fold CV Result")
    print("AUC\tACC\tRMSE")
    print("{:.5f}\t{:.5f}".format(test_auc, test_acc))

    print_args = dict()
    print_args["auc"] = round(test_auc, 4)
    print_args["auc_std"] = round(test_auc_std, 4)
    print_args["acc"] = round(test_acc, 4)
    print_args["acc_std"] = round(test_acc_std, 4)

    print_args['Model'] = model_name
    print_args['Dataset'] = data_name
    print_args.update(train_config)
    print_args.update(model_config)

    return print_args


def initialize_wandb(params_str):
    wandb.init(project="Camera-Invariant Domain Adaptation", entity="kwakjunyoung")
    wandb.run.name = params_str
    wandb.run.save()


def main(config):

    tm = localtime(time.time())
    params_str = f'{tm.tm_mon}_{tm.tm_mday}_{tm.tm_hour}:{tm.tm_min}:{tm.tm_sec}'

    if config.use_wandb:
        initialize_wandb(params_str)

    accelerator = Accelerator()
    device = accelerator.device

    model_name = config.model_name
    dataset_path = config.dataset_path
    data_name = config.data_name

    train_config = config.train_config
    batch_size = train_config.batch_size
    learning_rate = train_config.learning_rate
    optimizer = train_config.optimizer

    train_file_dir = './dataset/train_source.csv'
    train_dataset, train_data_loader = get_data_info(
        accelerator, train_file_dir, True, False, 4, batch_size, device=device)

    test_file_dir = './dataset/test.csv'
    test_dataset, test_data_loader = get_data_info(
        accelerator, test_file_dir, False, True, 4, batch_size, device=device)

    model_config, model = get_model_info(config, device, model_name)

    n_gpu = torch.cuda.device_count()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    if optimizer == "adam":
        opt = Adam(model.parameters(), lr=learning_rate)

    model, opt = accelerator.prepare(model, opt)

    result = model_train(model, device, opt, train_data_loader,
                         test_data_loader, config)

    submit = pd.read_csv('./dataset/sample_submission.csv')
    submit['mask_rle'] = result

    submit.to_csv(f'./baseline_submit.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="unet",
        help="The name of the model to train. \
            The default model is unet.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="train_source",
        help="The name of the dataset to use in training.",
    )
    parser.add_argument(
        "--negative_prob",
        type=float,
        default=1.0,
        help="reverse responses probability for hard negative pairs",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="dropout probability"
    )
    parser.add_argument(
        "--batch_size", type=float, default=16, help="Batch size"
    )
    parser.add_argument(
        "--describe", type=str, default="default", help="description of the training"
    )
    parser.add_argument(
        "--use_wandb", type=int, default=1
    )
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--optimizer", type=str,
                        default="adam", help="optimizer")

    parser.add_argument("--seed",  type=int, default=12405, help="seed")
    args = parser.parse_args()

    base_cfg_file = PathManager.open("configs/configuration.yaml", "r")
    base_cfg = yaml.safe_load(base_cfg_file)
    cfg = CN(base_cfg)
    cfg.set_new_allowed(True)

    cfg.model_name = args.model_name
    cfg.data_name = args.data_name
    cfg.use_wandb = args.use_wandb

    cfg.train_config.batch_size = int(args.batch_size)
    cfg.train_config.learning_rate = args.lr
    cfg.train_config.optimizer = args.optimizer
    cfg.train_config.describe = args.describe
    cfg.train_config.negative_prob = args.negative_prob
    cfg.train_config.dropout = args.dropout
    cfg.train_config.seed = args.seed

    if args.model_name == "unet":
        cfg.unet_config = cfg.unet_config[cfg.data_name]

    cfg.freeze()

    main(cfg)
