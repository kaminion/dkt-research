import os 
import argparse
import json
import pickle
import random

import numpy as np

import torch

from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adam
from torch.nn.functional import binary_cross_entropy, pad, one_hot
from sklearn import metrics 
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2012 import ASSIST2012
from data_loaders.ednet01 import EdNet01

# 모델 추가
from models.dkt_rear import DKT as DKT_REAR
from models.dkt_front import DKT as DKT_FRONT
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.dkvmn_text import SUBJ_DKVMN
from models.sakt import SAKT
from models.sakt_front import SAKT as SAKT_FRONT
from models.sakt_rear import SAKT as SAKT_REAR
from models.saint import SAINT
from models.saint_front import SAINT as SAINT_FRONT
from models.saint_rear import SAINT as SAINT_REAR
from models.auto import AUTO
from models.mekt import MEKT
from models.dirt import DeepIRT
from models.qakt import QAKT
from models.akt import AKT

# 모델에 따른 train
from models.dkt_rear import dkt_train as dk_rear_train
from models.dkt_front import dkt_train as dk_front_train
from models.dkt import dkt_train
from models.auto import auto_train
from models.dkvmn_text import train_model as plus_train
from models.sakt import sakt_train
from models.sakt_front import sakt_train as sakt_front_train
from models.sakt_rear import sakt_train as sakt_rear_train
from models.dkvmn import train_model as dkvmn_train
from models.saint import train_model as saint_train
from models.saint_front import train_model as saint_front_train
from models.saint_rear import train_model as saint_rear_train
from models.akt import train_model as akt_train

from models.utils import collate_fn, collate_ednet

# Cross Validation
from sklearn.model_selection import KFold

# wandb
import wandb

# seed 고정
seed = 42
#deterministic = True

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
#if deterministic:
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


# main program
def main(model_name, dataset_name, use_wandb):
    KAKAO_CKPTS = "/app/outputs/"
    if not os.path.isdir(f"{KAKAO_CKPTS}ckpts"): # original: ckpts
        os.mkdir(f"{KAKAO_CKPTS}ckpts")
    ckpt_path = os.path.join(f"{KAKAO_CKPTS}ckpts", model_name)
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
    
    if use_wandb == True:
        # wandb setting
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        wandb.init(project=f"{model_name}_{dataset_name}", config=train_config)

        # sweep config // optimization을 위한
        sweep_config = {
            'method': 'grid'
        }

        metric = {
            'name': 'loss',
            'goal': 'minimize'
        }


    # 데이터셋 추가 가능
    collate_pt = collate_fn
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2012":
        dataset = ASSIST2012(seq_len)
    elif dataset_name == "EDNET01":
        dataset = EdNet01(seq_len)
        collate_pt = collate_ednet

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
    train_model = None
    if model_name == "dkt":
        model = torch.nn.DataParallel(DKT(dataset.num_q, **model_config)).to(device)
        train_model = dkt_train
    elif model_name == "dkt-":
        model = torch.nn.DataParallel(DKT_FRONT(dataset.num_q, **model_config)).to(device)
        train_model = dk_front_train
    elif model_name == "dkt+":
        model = torch.nn.DataParallel(DKT_REAR(dataset.num_q, **model_config)).to(device)
        train_model = dk_rear_train
    elif model_name == 'dkvmn':
        model = torch.nn.DataParallel(DKVMN(dataset.num_q, **model_config)).to(device)
        train_model = dkvmn_train
    elif model_name == 'dkvmn+':
        model = torch.nn.DataParallel(SUBJ_DKVMN(dataset.num_q, num_qid=dataset.num_pid, **model_config)).to(device)
        train_model = plus_train
    elif model_name == 'dkvmn-':
        model = torch.nn.DataParallel(SUBJ_DKVMN(dataset.num_q, **model_config)).to(device)
        train_model = plus_train
    elif model_name == 'sakt':
        model = torch.nn.DataParallel(SAKT(dataset.num_q, **model_config)).to(device)
        train_model = sakt_train
    elif model_name == 'sakt-':
        model = torch.nn.DataParallel(SAKT_FRONT(dataset.num_q, **model_config)).to(device)
        train_model = sakt_front_train
    elif model_name == 'sakt+':
        model = torch.nn.DataParallel(SAKT_REAR(dataset.num_q, **model_config)).to(device)
        train_model = sakt_rear_train
    elif model_name == 'saint':
        model = torch.nn.DataParallel(SAINT(dataset.num_q, **model_config)).to(device)
        train_model = saint_train
    elif model_name == 'saint-':
        model = torch.nn.DataParallel(SAINT_FRONT(dataset.num_q, **model_config)).to(device)
        train_model = saint_front_train
    elif model_name == "saint+":
        model = torch.nn.DataParallel(SAINT_REAR(dataset.num_q, **model_config)).to(device)
        train_model = saint_rear_train
    elif model_name == 'akt':
        model = torch.nn.DataParallel(AKT(n_question=dataset.num_q, n_pid=dataset.num_pid, **model_config)).to(device)
        train_model = akt_train
    elif model_name == "auto":
        model = torch.nn.DataParallel(AUTO(num_q=dataset.num_q, **model_config)).to(device)
        train_model = auto_train
    elif model_name == "mekt":
        model = MEKT(dataset.num_q, **model_config).to(device)
    elif model_name == "dirt":
        model = DeepIRT(dataset.num_q, dataset.num_u, **model_config).to(device)
    elif model_name == "qakt":
        model = QAKT(dataset.num_q, **model_config).to(device)

    else: 
        print(f"The wrong model name was used...: {model_name}")
        return
    
    # 데이터셋 분할
    data_size = len(dataset)
    train_size = int(data_size * train_ratio) 
    valid_size = int(data_size * ((1.0 - train_ratio) / 2.0))
    test_size = data_size - train_size - valid_size

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size], generator=torch.Generator(device=device)
    )

    # pickle에 얼마만큼 분할했는지 저장
    if os.path.exists(os.path.join(dataset.dataset_dir, "train_indices.pkl")):
        with open(
            os.path.join(dataset.dataset_dir, "train_indices.pkl"), "rb"
        ) as f:
            train_dataset.indices = pickle.load(f)
        with open(
            os.path.join(dataset.dataset_dir, "valid_indicies.pkl"), "rb"
        ) as f:
            valid_dataset.indices = pickle.load(f)
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
            os.path.join(dataset.dataset_dir, "valid_indicies.pkl"), "wb"
        ) as f:
            pickle.dump(valid_dataset.indices, f)
        with open(
            os.path.join(dataset.dataset_dir, "test_indices.pkl"), "wb"
        ) as f:
            pickle.dump(test_dataset.indices, f)

    # Loader에 데이터 적재
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pt, generator=torch.Generator(device=device)
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pt, generator=torch.Generator(device=device)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_pt, generator=torch.Generator(device=device)
    )

    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    opt.lr_scheduler = lr_scheduler

    # 모델에서 미리 정의한 함수로 AUCS와 LOSS 계산    
    aucs, loss_means, accs, q_accs, q_cnts, precisions, recalls, f1s = \
        train_model(
            model, train_loader, valid_loader, test_loader, dataset.num_q, num_epochs, batch_size, opt, ckpt_path
        )
    # DKT나 다른 모델 학습용
    # aucs, loss_means = model.train_model(train_loader, test_loader, num_epochs, opt, ckpt_path)
    
    with open(os.path.join(ckpt_path, f"aucs_{seed}.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, f"loss_means_{seed}.pkl"), "wb") as f:
        pickle.dump(loss_means, f)
    with open(os.path.join(ckpt_path, f"accs_{seed}.pkl"), "wb") as f:
        pickle.dump(accs, f)
    with open(os.path.join(ckpt_path, f"q_accs_{seed}.pkl"), "wb") as f:
        pickle.dump(q_accs, f)
    with open(os.path.join(ckpt_path, f"q_cnts_{seed}.pkl"), "wb") as f:
        pickle.dump(q_cnts, f)
        
    # precisions, recalls, f1s
    with open(os.path.join(ckpt_path, f"precisions_{seed}.pkl"), "wb") as f:
        pickle.dump(precisions, f)
    with open(os.path.join(ckpt_path, f"recalls_{seed}.pkl"), "wb") as f:
        pickle.dump(recalls, f)
    with open(os.path.join(ckpt_path, f"f1s_{seed}.pkl"), "wb") as f:
        pickle.dump(f1s, f)
        
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

    # true_list 에 없으면 자동적으로 false가 됨.
    true_list = ['true', 'yes', '1', 't', 'y']

    parser.add_argument(
        '--use_wandb',
        type= lambda s: s.lower() in true_list,
        default=False,
        help="This option value is using wandb"
    )

    args = parser.parse_args()

    main(args.model_name, args.dataset_name, args.use_wandb)