import os 
import argparse
import json
import pickle

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
from models.dkt import DKT
from models.dkvmn import DKVMN
from models.dkvmn_text import SUBJ_DKVMN
from models.sakt import SAKT
from models.saint import SAINT
from models.auto import AUTO
from models.mekt import MEKT
from models.dirt import DeepIRT
from models.qakt import QAKT
from models.akt import AKT

# 모델에 따른 train
from models.dkt import dkt_train
from models.auto import auto_train
from models.dkvmn_text import train_model as plus_train

from models.utils import collate_fn, collate_ednet

# Cross Validation
from sklearn.model_selection import KFold

# wandb
import wandb

def train_model(model, train_loader, test_loader, exp_loader, num_q, num_epochs, opt, ckpt_path):
    '''
        Args:
            train_loader: the PyTorch DataLoader instance for training
            test_loader: the PyTorch DataLoader instance for test
            num_epochs: the number of epochs
            opt: the optimization to train this model
            ckpt_path: the path to save this model's parameters
    '''
    aucs = []
    loss_means = []  

    max_auc = 0

    for i in range(0, num_epochs):
        loss_mean = []
        

        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            model.train()

            # 현재까지의 입력을 받은 뒤 다음 문제 예측

            # SAINT LOSS DKT DKVMN
            y = model(q.long(), r.long())
            # SAKT LOSS
            # y, _ = model(q.long(), r.long(), qshft_seqs.long())
            y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)
            # DKVMN+ LOSS  at_s, at_t, at_m, q2diff
            # y, _ = model(q.long(), r.long(), bert_s, bert_t, bert_m, q2diff_seqs.long())

            # AKT LOSS
            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            opt.zero_grad()
            # y, akt_loss = model(q.long(), (q + r).long(), r.long(), pid_seqs.long()) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
            y = torch.masked_select(y, m)
            t = torch.masked_select(rshft_seqs, m)
            h = torch.masked_select(hint_seqs, m)

            loss = binary_cross_entropy(y, t) 
            # loss += akt_loss 
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            for data in test_loader:
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()

                # AKT LOSS
                # y, _ = model(q.long(), (q + r).long(), r.long(), pid_seqs.long()) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy

                # y, _ = model(q.long(), r.long(), bert_s, bert_t, bert_m, q2diff_seqs.long())
                
                # SAINT DKT DKVMN
                y = model(q.long(), r.long())
                # SAKT LOSS
                # y, _ = model(q.long(), r.long(), qshft_seqs.long())
                y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft_seqs, m).detach().cpu()
                h = torch.masked_select(hint_seqs, m).detach().cpu()

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )

                loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                
                # print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean} ")

                if auc > max_auc : 
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    print(f"Epoch {i}, previous AUC: {max_auc}, max AUC: {auc}")
                    max_auc = auc

                # aucs.append(auc)
                loss_means.append(loss_mean)

    # 실제 성능측정
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    for i in range(1, num_epochs + 1):
        with torch.no_grad():
            for data in exp_loader:
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()
                # DKT SAINT DKVMN
                y = model(q.long(), r.long())
                # AKT LOSS
                # y, _ = model(q.long(), (q + r).long(), r.long(), pid_seqs.long()) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
                y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

                # SAINT LOSS
                # y, _ = model(q.long(), r.long(), qshft_seqs.long())
                # y = (y * one_hot(qshft_seqs.long(), num_q)).sum(-1)

                # y, _ = model(q.long(), r.long(), bert_s, bert_t, bert_m, q2diff_seqs.long())

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(rshft_seqs, m).detach().cpu()
                h = torch.masked_select(hint_seqs, m).detach().cpu()

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )

                loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                
                print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean}")

                aucs.append(auc)
                # loss_means.append(loss_mean)

    return aucs, loss_means

# main program
def main(model_name, dataset_name, use_wandb):
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
    elif model_name == 'dkvmn':
        model = torch.nn.DataParallel(DKVMN(dataset.num_q, **model_config)).to(device)
    elif model_name == 'dkvmn+':
        model = torch.nn.DataParallel(SUBJ_DKVMN(dataset.num_q, num_qid=dataset.num_pid, **model_config)).to(device)
        train_model = plus_train
    elif model_name == 'dkvmn-':
        model = torch.nn.DataParallel(SUBJ_DKVMN(dataset.num_q, **model_config)).to(device)
        train_model = plus_train
    elif model_name == 'sakt':
        model = torch.nn.DataParallel(SAKT(dataset.num_q, **model_config)).to(device)
    elif model_name == 'saint':
        model = torch.nn.DataParallel(SAINT(dataset.num_q, **model_config)).to(device)
    elif model_name == 'akt':
        model = torch.nn.DataParallel(AKT(n_question=dataset.num_q, n_pid=dataset.num_pid, **model_config)).to(device)
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
        print("The wrong model name was used...")
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
    aucs, loss_means, accs, q_accs, q_cnts = \
        train_model(
            model, train_loader, valid_loader, test_loader, dataset.num_q, num_epochs, batch_size, opt, ckpt_path
        )
    # DKT나 다른 모델 학습용
    # aucs, loss_means = model.train_model(train_loader, test_loader, num_epochs, opt, ckpt_path)
    
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)
    with open(os.path.join(ckpt_path, "accs.pkl"), "wb") as f:
        pickle.dump(accs, f)
    with open(os.path.join(ckpt_path, "q_accs.pkl"), "wb") as f:
        pickle.dump(q_accs, f)
    with open(os.path.join(ckpt_path, "q_cnts.pkl"), "wb") as f:
        pickle.dump(q_cnts, f)
        
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