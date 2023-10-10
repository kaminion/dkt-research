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
# from models.dkt_sc import dkt_train as dkf_train
# from models.dkt_rear import dkt_train as dk_rear_train
# from models.dkt_front import dkt_train as dk_front_train
# from models.dkt import dkt_train
# from models.auto import auto_train
# from models.dkvmn_text import train_model as plus_train
# from models.dkvmn_back import train_model as minus_train
# from models.sakt import sakt_train
# from models.sakt_front import sakt_train as sakt_front_train
# from models.sakt_rear import sakt_train as sakt_rear_train
# from models.dkvmn import train_model as dkvmn_train
# from models.saint import train_model as saint_train
# from models.saint_front import train_model as saint_front_train
# from models.saint_rear import train_model as saint_rear_train
# from models.akt import train_model as akt_train

from models.utils import collate_fn, collate_ednet, cal_acc_class

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
    
# Train function
def train_model(model, train_loader, valid_loader, num_q, num_epochs, opt, ckpt_path, mode=0, wandb=None):
    max_auc = 0
        
    for epoch in range(0, num_epochs):
        auc_mean = []
        loss_mean = []
        acc_mean = []

        for i, data in enumerate(train_loader, 0):
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            model.train()
            
            # 현재 답안 예측
            inpt_q = q.long()
            pred_t = r
            if mode == 1: # 다음 답안 예측
                inpt_q = qshft_seqs.long()
                pred_t = rshft_seqs
            elif mode == 2: # 스코어 예측
                pred_t = pid_seqs
            elif mode == 3: # 다음 스코어 예측
                inpt_q = qshft_seqs.long()
                pred_t = pidshift

            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y = model(q.long(), r.long(), bert_s, bert_t, bert_m) # sakt는 qshft_seqs.long() 추가
            y = (y * one_hot(inpt_q, num_q)).sum(-1)

            opt.zero_grad()
            y = torch.masked_select(y, m)
            t = torch.masked_select(pred_t, m) # rshft 대신 pidshift
            h = torch.masked_select(hint_seqs, m)

            loss = binary_cross_entropy(y, t) 
            loss.backward()
            opt.step()

            loss_mean.append(loss.detach().cpu().numpy())
            auc_mean.append(metrics.roc_auc_score(
                y_true=t.detach().cpu().numpy(), y_score=y.detach().cpu().numpy()
            ))
            bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
            acc_mean.append(metrics.accuracy_score(t.detach().cpu().numpy(), bin_y))
            
        
        loss_mean = np.mean(loss_mean)
        auc_mean = np.mean(auc_mean)
        acc_mean = np.mean(acc_mean)
        
        if(wandb != None):
            wandb.log(
            {
                "epoch": epoch,
                "train_auc": auc_mean, 
                "train_acc": acc_mean,
                "train_loss": loss_mean
            })

        print(f"[Train] Epoch: {epoch}, AUC: {auc_mean}, acc: {acc_mean}, Loss Mean: {np.mean(loss_mean)}")

    with torch.no_grad():
        auc_mean = []
        loss_mean = []
        acc_mean = []
        
        best_loss = 10 ** 9
        patience_limit = 3
        patience_check = 0
        
        for i, data in enumerate(valid_loader):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

            # 현재 답안 예측
            inpt_q = q.long()
            pred_t = r
            if mode == 1: # 다음 답안 예측
                inpt_q = qshft_seqs.long()
                pred_t = rshft_seqs
            elif mode == 2: # 스코어 예측
                pred_t = pid_seqs
            elif mode == 3: # 다음 스코어 예측
                inpt_q = qshft_seqs.long()
                pred_t = pidshift

            model.eval()
            
            y = model(q.long(), r.long(), bert_s, bert_t, bert_m) # sakt는 qshft_seqs.long() 추가
            y = (y * one_hot(inpt_q, num_q)).sum(-1)

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(pred_t, m).detach().cpu()
            h = torch.masked_select(hint_seqs, m).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            auc_mean.append(auc)
            
            bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
            acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)
            acc_mean.append(acc)

            loss = binary_cross_entropy(y, t)
            loss_mean.append(loss)
            
            if loss > best_loss:
                patience_check += 1
                
                if patience_check >= patience_limit:
                    break
            else:
                best_loss = loss
                patience_check = 0
            
            if auc > max_auc : 
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        ckpt_path, "model.ckpt"
                    )
                )
                max_auc = auc
                
        loss_mean = np.mean(loss_mean)
        auc_mean = np.mean(auc_mean)
        acc_mean = np.mean(acc_mean)
        
        if(wandb != None):
            wandb.log(
            {
                "epoch": epoch,
                "val_auc": auc_mean, 
                "val_acc": acc_mean,
                "val_loss": loss_mean
            })
        print(f"[Valid] {epoch} Result: AUC: {auc_mean}, ACC: {acc_mean}, loss: {loss_mean}")

        print(f"========== Finished Epoch: {epoch} ============")


# Test function
def test_model(model, test_loader, num_q, ckpt_path, mode=0):
    
    wandb.init(group=f"test_{date.today().isoformat()}_{mode}", name=f"{mode}", reinit=True)
    
    # 실제 성능측정, mode 0은 현재 답안 예측, 1은 다음 답안 예측, 2는 스코어 예측
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    loss_mean = []
    aucs = []
    accs = []
    precisions = []
    recalls = []
    f1s = []
    
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            
            # 현재 답안 예측
            inpt_q = q.long()
            pred_t = r
            if mode == 1: # 다음 답안 예측
                inpt_q = qshft_seqs.long()
                pred_t = rshft_seqs
            elif mode == 2: # 스코어 예측
                pred_t = pid_seqs
            elif mode == 3: # 다음 스코어 예측
                inpt_q = qshft_seqs.long()
                pred_t = pidshift
                
            model.eval()
            y = model(q.long(), r.long(), bert_s, bert_t, bert_m)
            y = (y * one_hot(inpt_q, num_q)).sum(-1)

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            q = torch.masked_select(q, m).detach().cpu()
            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(pred_t, m).detach().cpu()
            h = torch.masked_select(hint_seqs, m).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
            acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)
            precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
            recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
            f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
            loss = binary_cross_entropy(y, t) 
            
            print(f"[Test] number: {i}, AUC: {auc}, ACC: {acc}, loss: {loss}")

            aucs.append(auc)
            loss_mean.append(loss)
            accs.append(acc)
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
        loss_means = np.mean(loss_mean) # 실제 로스 평균값을 구함
        auc_mean = np.mean(aucs)
        acc_mean = np.mean(accs)
        f1_mean = np.mean(f1s)
        
        if(wandb != None):
            wandb.log(
            {
                "test_auc": auc_mean, 
                "test_acc": acc_mean,
                "test_loss": loss_mean
            })
            wandb.finish()
        
        print(f"[Total Test]: AUC: {auc_mean}, ACC: {acc_mean}, F1 Score: {f1_mean} ")

    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s


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
    collate_pt = collate_fn
    if dataset_name == "ASSIST2009":
        dataset = ASSIST2009(seq_len)
    elif dataset_name == "ASSIST2012":
        dataset = ASSIST2012(seq_len)
    elif dataset_name == "EDNET01":
        dataset = EdNet01(seq_len)
        collate_pt = collate_ednet
    elif dataset_name == "CSEDM":
        dataset = CSEDM(seq_len)

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
    # train_model = None
    if model_name == "dkt":
        model = torch.nn.DataParallel(DKT(dataset.num_q, **model_config)).to(device)
    elif model_name == "dkf":
        model = torch.nn.DataParallel(DKT_F(dataset.num_q, **model_config)).to(device)
    elif model_name == "dkt-":
        model = torch.nn.DataParallel(DKT_FRONT(dataset.num_q, **model_config)).to(device)
    elif model_name == "dkt+":
        model = torch.nn.DataParallel(DKT_REAR(dataset.num_q, **model_config)).to(device)
    elif model_name == 'dkvmn':
        model = torch.nn.DataParallel(DKVMN(dataset.num_q, **model_config)).to(device)
    elif model_name == 'dkvmn+':
        model = torch.nn.DataParallel(SUBJ_DKVMN(dataset.num_q, num_qid=dataset.num_pid, **model_config)).to(device)
    elif model_name == 'dkvmn-':
        model = torch.nn.DataParallel(BACK_DKVMN(dataset.num_q, **model_config)).to(device)
    elif model_name == 'sakt':
        model = torch.nn.DataParallel(SAKT(dataset.num_q, **model_config)).to(device)
    elif model_name == 'sakt-':
        model = torch.nn.DataParallel(SAKT_FRONT(dataset.num_q, **model_config)).to(device)
    elif model_name == 'sakt+':
        model = torch.nn.DataParallel(SAKT_REAR(dataset.num_q, **model_config)).to(device)
    elif model_name == 'saint':
        model = torch.nn.DataParallel(SAINT(dataset.num_q, **model_config)).to(device)
    elif model_name == 'saint-':
        model = torch.nn.DataParallel(SAINT_FRONT(dataset.num_q, **model_config)).to(device)
    elif model_name == "saint+":
        model = torch.nn.DataParallel(SAINT_REAR(dataset.num_q, **model_config)).to(device)
    elif model_name == 'akt':
        model = torch.nn.DataParallel(AKT(n_question=dataset.num_q, n_pid=dataset.num_pid, **model_config)).to(device)
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
    
    # 현재꺼 예측, 다음꺼 예측, CSEDM 현재꺼 예측, 다음꺼 예측 이렇게 4개로 디자인
    pred_next = ["dkt", "dkt+", "dkt-", "sakt", "sakt+", "sakt-"]
    pred_now  = ["dkvmn", "dkvmn+", "dkvmn-", "akt", "saint", "saint+", "saint-"]
    
    mode = 0
    if model_name in pred_next:
        mode = 1
    elif model_name in pred_now and dataset_name == "CSEDM":
        mode = 2
    elif model_name in pred_next and dataset_name == "CSEDM":
        mode = 3
    
    # IIFE 즉시 실행 함수로 패킹해서 wandb로 넘겨줌
    def train_main():
        proj_name = f"{model_name}_{dataset_name}"
        num_epochs = train_config["num_epochs"]
        kfold = KFold(n_splits=5, shuffle=True)
        cv_name = f"{wandb.util.generate_id()}"

        for fold, (train_ids, valid_ids) in enumerate(kfold.split(tv_dataset)):
            fold += 1
            print(f"========={fold}==========")
            
            if use_wandb == True:
                run_name = f"{date.today().isoformat()}-{cv_name}-{fold:02}-runs"
                run = wandb.init(group=f"cv_{cv_name}_{fold}", name=run_name, reinit=True)
                
                assert run is not None
                assert type(run) is wandb.sdk.wandb_run.Run
                wandb.summary["cv_fold"] = fold
                wandb.summary["num_cv_folds"] = kfold.n_splits
                wandb.summary["cv_random_state"] = kfold.random_state
                
                num_epochs = wandb.config.epochs
                opt.param_groups[0]['lr'] = wandb.config.learning_rate
                model.hidden_size = wandb.config.hidden_size
            
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
            train_model(
                model, train_loader, valid_loader, dataset.num_q, num_epochs, opt, ckpt_path, mode, wandb
            )
            
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
                'epochs': {'values': [100]},
                'learning_rate': {'values': [1e-2, 1e-3]},
                'hidden_size': {'values': [50, 100]}
            }
        }
        
        sweep_id = wandb.sweep(sweep=sweep_config, project=proj_name)
        wandb.agent(sweep_id, function=train_main, project=proj_name)
    else:
        train_main()


    # 마지막 테스트
    test_loader = DataLoader(
    test_dataset, batch_size=batch_size,
    collate_fn=collate_pt, generator=torch.Generator(device=device),
    )
    
    auc, loss_mean, acc, q_acc, q_cnt, precision, recall, f1 = \
    test_model(
        model, test_loader, dataset.num_q, ckpt_path, mode
    )

    with open(os.path.join(ckpt_path, f"aucs_{seed}.pkl"), "wb") as f:
        pickle.dump(auc, f)
    with open(os.path.join(ckpt_path, f"loss_means_{seed}.pkl"), "wb") as f:
        pickle.dump(loss_mean, f)
    with open(os.path.join(ckpt_path, f"accs_{seed}.pkl"), "wb") as f:
        pickle.dump(acc, f)
    with open(os.path.join(ckpt_path, f"q_accs_{seed}.pkl"), "wb") as f:
        pickle.dump(q_acc, f)
    with open(os.path.join(ckpt_path, f"q_cnts_{seed}.pkl"), "wb") as f:
        pickle.dump(q_cnt, f)
    
    # precisions, recalls, f1s
    with open(os.path.join(ckpt_path, f"precisions_{seed}.pkl"), "wb") as f:
        pickle.dump(precision, f)
    with open(os.path.join(ckpt_path, f"recalls_{seed}.pkl"), "wb") as f:
        pickle.dump(recall, f)
    with open(os.path.join(ckpt_path, f"f1s_{seed}.pkl"), "wb") as f:
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