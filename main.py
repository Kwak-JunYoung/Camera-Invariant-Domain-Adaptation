from models.unet import UNet
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from data_loaders import CustomDataset
import numpy as np
import wandb
import torch
import random 

# Random seed
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True

# Obtain information of the model
def get_model_info(config, model_name):
    if model_name == "unet":
        model_config = config["models"]["unet"]
        model = UNet()
    else: 
        raise NotImplementedError("model name is not valid")
    return model_config, model

def get_data_loader(accelerator):
    accelerator.prepare()


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
    wandb.init(project="CLinKT", entity="kwakjunyoung")
    wandb.run.name = params_str
    wandb.run.save()

def main():
    dataset = CustomDataset(csv_file='./train_source.csv')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)