import os 
import argparse
import json
import pickle
import random
from datetime import date

import numpy as np

import torch

from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.optim import SGD, Adam
from torch.nn.functional import binary_cross_entropy, pad, one_hot
from sklearn import metrics 
from data_loaders.assist2009 import ASSIST2009
from data_loaders.assist2012 import ASSIST2012
from data_loaders.ednet01 import EdNet01
from data_loaders.csedm import CSEDM

# 모델 추가
from models.dkt_sc import DKT_FUSION as DKT_F
from models.dkt_rear import DKT as DKT_REAR
from models.dkt_front import DKT as DKT_FRONT
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.dkvmn_text import SUBJ_DKVMN
from models.dkvmn_back import BACK_DKVMN
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
from models.dkt_sc import dkt_train as dkf_train
from models.dkt_rear import dkt_train as dk_rear_train
from models.dkt_front import dkt_train as dk_front_train
from models.dkt import train_model as dkt_train
from models.auto import auto_train
from models.dkvmn_text import train_model as plus_train
from models.dkvmn_back import train_model as minus_train
from models.sakt import train_model as sakt_train
from models.sakt_front import sakt_train as sakt_front_train
from models.sakt_rear import sakt_train as sakt_rear_train
from models.dkvmn import train_model as dkvmn_train
from models.saint import train_model as saint_train
from models.saint_front import train_model as saint_front_train
from models.saint_rear import train_model as saint_rear_train
from models.akt import train_model as akt_train

# 모델에 따른 test
from models.dkt import test_model as dkt_test
from models.dkvmn import test_model as dkvmn_test
from models.dkvmn_text import test_model as plus_test
from models.sakt import test_model as sakt_test
from models.saint import test_model as saint_test
from models.akt import test_model as akt_test

from models.utils import collate_fn, collate_csedm, cal_acc_class, reset_weight

# Cross Validation
from sklearn.model_selection import KFold

# wandb
import wandb

# seed 고정
seed = 3407
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
    # KAKAO_CKPTS = "./"
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
    
    # 데이터셋 추가 가능
    mode = 0
    collate_pt = collate_fn
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2012":
        dataset = ASSIST2012(seq_len)
    elif dataset_name == "EDNET01":
        dataset = EdNet01(seq_len)
    elif dataset_name == "CSEDM":
        mode = 1
        dataset = CSEDM(seq_len)
        collate_pt = collate_csedm

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
    test_model = None
    if model_name == "dkt":
        model = torch.nn.DataParallel(DKT(dataset.num_q, **model_config)).to(device)
        train_model = dkt_train
        test_model = dkt_test
    elif model_name == "dkf":
        model = torch.nn.DataParallel(DKT_F(dataset.num_q, **model_config)).to(device)
    elif model_name == "dkt-":
        model = torch.nn.DataParallel(DKT_FRONT(dataset.num_q, **model_config)).to(device)
    elif model_name == "dkt+":
        model = torch.nn.DataParallel(DKT_REAR(dataset.num_q, **model_config)).to(device)
    elif model_name == 'dkvmn':
        model = torch.nn.DataParallel(DKVMN(dataset.num_q, **model_config)).to(device)
        train_model = dkvmn_train
        test_model = dkvmn_test
    elif model_name == 'dkvmn+':
        model = torch.nn.DataParallel(SUBJ_DKVMN(dataset.num_q, num_qid=dataset.num_pid, **model_config)).to(device)
        train_model = plus_train
        test_model = plus_test
    elif model_name == 'dkvmn-':
        model = torch.nn.DataParallel(BACK_DKVMN(dataset.num_q, **model_config)).to(device)
    elif model_name == 'sakt':
        model = torch.nn.DataParallel(SAKT(dataset.num_q, **model_config)).to(device)
        train_model = sakt_train 
        test_model = sakt_test
    elif model_name == 'sakt-':
        model = torch.nn.DataParallel(SAKT_FRONT(dataset.num_q, **model_config)).to(device)
    elif model_name == 'sakt+':
        model = torch.nn.DataParallel(SAKT_REAR(dataset.num_q, **model_config)).to(device)
    elif model_name == 'saint':
        model = torch.nn.DataParallel(SAINT(dataset.num_q, **model_config)).to(device)
        train_model = saint_train
        test_model = saint_test
    elif model_name == 'saint-':
        model = torch.nn.DataParallel(SAINT_FRONT(dataset.num_q, **model_config)).to(device)
    elif model_name == "saint+":
        model = torch.nn.DataParallel(SAINT_REAR(dataset.num_q, **model_config)).to(device)
    elif model_name == 'akt':
        model = torch.nn.DataParallel(AKT(n_question=dataset.num_q, n_pid=dataset.num_pid, **model_config)).to(device)
        train_model = akt_train
        test_model = akt_test
    elif model_name == "auto":
        model = torch.nn.DataParallel(AUTO(num_q=dataset.num_q, **model_config)).to(device)
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
    
    # 연결
    tv_dataset = ConcatDataset([train_dataset, valid_dataset])

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
        
    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.5)
    opt.lr_scheduler = lr_scheduler
    
    # IIFE 즉시 실행 함수로 패킹해서 wandb로 넘겨줌
    def train_main():
        proj_name = f"{model_name}_{dataset_name}"
        num_epochs = train_config["num_epochs"]
        kfold = KFold(n_splits=5, shuffle=True)

        for fold, (train_ids, valid_ids) in enumerate(kfold.split(tv_dataset)):
            fold += 1
            print(f"========={fold}==========")
            model.apply(reset_weight)
            
            cv_name = f"{wandb.util.generate_id()}"
            run_name = f"{date.today().isoformat()}-{cv_name}-{fold:02}-runs"
            run = wandb.init(group=f"cv_{cv_name}_{fold}", name=run_name, reinit=True)
            
            seed = wandb.config.seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            
            assert run is not None
            assert type(run) is wandb.sdk.wandb_run.Run
            wandb.summary["cv_fold"] = fold
            wandb.summary["num_cv_folds"] = kfold.n_splits
            wandb.summary["cv_random_state"] = kfold.random_state
            
            num_epochs = wandb.config.epochs
            opt.param_groups[0]['lr'] = wandb.config.learning_rate
            
            # 모델 파라미터
            # model.hidden_size = wandb.config.hidden_size
            model.dropout = wandb.config.dropout
            
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_ids)

            # Loader에 데이터 적재
            train_loader = DataLoader(
                tv_dataset, batch_size=batch_size,
                collate_fn=collate_pt, generator=torch.Generator(device=device),
                sampler=train_subsampler
            )
            valid_loader = DataLoader(
                tv_dataset, batch_size=batch_size,
                collate_fn=collate_pt, generator=torch.Generator(device=device),
                sampler=valid_subsampler
            )

            # 모델에서 미리 정의한 함수로 AUCS와 LOSS 계산    
            # auc, loss_mean, acc, q_acc, q_cnt, precision, recall, f1 = \
            aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s = train_model(
                model, train_loader, valid_loader, dataset.num_q, num_epochs, opt, ckpt_path, mode, use_wandb
            )
            
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_aucs_{seed}.pkl"), "wb") as f:
                pickle.dump(aucs, f)
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_loss_means_{seed}.pkl"), "wb") as f:
                pickle.dump(loss_means, f)
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_accs_{seed}.pkl"), "wb") as f:
                pickle.dump(accs, f)
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_q_accs_{seed}.pkl"), "wb") as f:
                pickle.dump(q_accs, f)
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_q_cnts_{seed}.pkl"), "wb") as f:
                pickle.dump(cnt, f)
            # precisions, recalls, f1s
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_precisions_{seed}.pkl"), "wb") as f:
                pickle.dump(precisions, f)
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_recalls_{seed}.pkl"), "wb") as f:
                pickle.dump(recalls, f)
            with open(os.path.join(ckpt_path, f"{fold}_{seq_len}_f1s_{seed}.pkl"), "wb") as f:
                pickle.dump(f1s, f)
            
            wandb.finish()
            
            # DKT나 다른 모델 학습용
            # aucs, loss_means = model.train_model(train_loader, test_loader, num_epochs, opt, ckpt_path)
    
    if use_wandb == True:
        # wandb setting
        os.environ['WANDB_API_KEY'] = WANDB_API_KEY
        proj_name = f"{model_name}_{dataset_name}"

        # sweep config // optimization을 위한
        sweep_config = {
            'method': 'grid',
            'name': f'DKT-{model_name}',
            'metric': {
                'name': 'val_auc',
                'goal': 'maximize'
            },
            'parameters': {
                'epochs': {'values': [300]},
                'seed': {'values': [3407]},
                'dropout': {'values': [0, 0.05, 0.1, 0.15, 0.2, 0.25]},
                'learning_rate': {'values': [5*1e-6, 1e-5, 1e-4]},
                # 'dim_s': {'values': [20, 50]},
                # 'size_m': {'values': [20, 50]}
                # 'emb_size': {'values': [256, 512]},
                # 'hidden_size': {'values': [256, 512]}
            }
        }
        
        sweep_id = wandb.sweep(sweep=sweep_config, project=proj_name)
        wandb.agent(sweep_id, function=train_main, project=proj_name)
    else: 
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size,
            collate_fn=collate_pt, generator=torch.Generator(device=device)
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size,
            collate_fn=collate_pt, generator=torch.Generator(device=device)
        )
        train_model(
            model, train_loader, valid_loader, dataset.num_q, num_epochs, opt, ckpt_path, mode, use_wandb
        )

    # 마지막 테스트
    test_loader = DataLoader(
    test_dataset, batch_size=batch_size,
    collate_fn=collate_pt, generator=torch.Generator(device=device),
    )
    
    auc, loss_mean, acc, q_acc, q_cnt, precision, recall, f1 = \
    test_model(
        model, test_loader, dataset.num_q, ckpt_path, mode, use_wandb
    )

    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_aucs_{seed}.pkl"), "wb") as f:
        pickle.dump(auc, f)
    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_loss_means_{seed}.pkl"), "wb") as f:
        pickle.dump(loss_mean, f)
    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_accs_{seed}.pkl"), "wb") as f:
        pickle.dump(acc, f)
    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_q_accs_{seed}.pkl"), "wb") as f:
        pickle.dump(q_acc, f)
    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_q_cnts_{seed}.pkl"), "wb") as f:
        pickle.dump(q_cnt, f)
    
    # precisions, recalls, f1s
    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_precisions_{seed}.pkl"), "wb") as f:
        pickle.dump(precision, f)
    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_recalls_{seed}.pkl"), "wb") as f:
        pickle.dump(recall, f)
    with open(os.path.join(ckpt_path, f"{optimizer}_{seq_len}_f1s_{seed}.pkl"), "wb") as f:
        pickle.dump(f1, f)
        
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