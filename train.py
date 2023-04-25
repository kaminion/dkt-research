import os 
import argparse
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam

from data_loaders.assist2009 import ASSIST2009

from models.dkt import DKT
from models.dkvmn import DKVMN
from models.clkt import CLKT
from models.mekt import MEKT
from models.dirt import DeepIRT
from models.qakt import QAKT
from models.utils import collate_fn

# wandb
import wandb



# main program
def main(model_name, dataset_name):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")
    
    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    
    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)
    
    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    with open("wandb_config.json") as f:
        wandb_config = json.load(f)
        WANDB_API_KEY = wandb_config['wandb_api_key']
    
    batch_size = train_config["batch_size"]
    num_epochs = train_config["num_epochs"]
    train_ratio = train_config["train_ratio"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"] # can be sgd, adam
    seq_len = train_config["seq_len"] # 샘플링 할 갯수

    # wandb setting
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    wandb.init(config=train_config)


    # 데이터셋 추가 가능
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # JSON으로 설정값 저장
    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    
    ## 가변 벡터이므로 **
    if model_name == "dkt":
        model = DKT(dataset.num_q, **model_config).to(device)
    elif model_name == 'dkvmn':
        model = DKVMN(dataset.num_q, **model_config).to(device)
    elif model_name == "clkt":
        model = CLKT(dataset.num_q, **model_config).to(device)
    elif model_name == "mekt":
        model = MEKT(dataset.num_q, **model_config).to(device)
    elif model_name == "dirt":
        model = DeepIRT(dataset.num_q, dataset.num_u, **model_config).to(device)
    elif model_name == "qakt":
        model = QAKT(dataset.num_q, **model_config).to(device)

    else: 
        print("The wrong model name was used...")
        return
    
    # 데이터셋 분할
    train_size = int(len(dataset) * train_ratio) 
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(
        dataset, [train_size, test_size]
    )

    # pickle에 얼마만큼 분할했는지 저장
    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(
            os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
        ) as f:
            train_dataset.indices = pickle.load(f)
        with open(
            os.path.join(dataset.dataset_dir, "test_indices.pkl"), "rb"
        ) as f:
            test_dataset.indices = pickle.load(f)
    else:
        with open(
            os.path.join(dataset.dataset_dir, "train_indices.pkl"), "wb"
        ) as f:
            pickle.dump(train_dataset.indices, f)
        with open(
            os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
        ) as f:
            pickle.dump(test_dataset.indices, f)

    # Loader에 데이터 적재
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn
    )

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    opt.lr_scheduler = lr_scheduler

    # 모델에서 미리 정의한 함수로 AUCS와 LOSS 계산    
    aucs, loss_means = \
        model.train_model(
            train_loader, test_loader, num_epochs, opt, ckpt_path
        )
    
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)

# program main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="dkt",
        help="The name of the model to train. \
            The possible models are in [dkt]. \
            The default model is dkt."
    )
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="ASSIST2009",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [ASSIST2009]. \
            The default dataset is ASSIST2009."
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name)