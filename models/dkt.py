import os
import time
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, MultiheadAttention, LayerNorm
from models.emb import STFTEmbedding
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from models.utils import save_auc, log_auc, common_append, val_append, mean_eval, mean_eval_ext, dkt_train, dkt_test, early_stopping

from transformers import BertModel, BertConfig, DistilBertConfig, DistilBertModel
from models.utils import cal_acc_class

import wandb

class DKT(Module):
    '''
        Args: 
            num_q: total number of questions in given the dataset
    '''
    def __init__(self, num_q, emb_size, hidden_size, dropout=0.2):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        
        # BERT for feature extraction
        # bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        # self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        # distilconfig = DistilBertConfig(output_hidden_states=True)
        # self.bertmodel = DistilBertModel.from_pretrained('distilbert-base-uncased', config=distilconfig)
        # self.at_emb_layer = Linear(768, self.emb_size)
        # self.at2_emb_layer = Linear(512, self.emb_size)
        
        # self.v_emb_layer = Linear(self.emb_size * 2, self.hidden_size)
        
        # self.e_layer = Linear(self.hidden_size, self.emb_size)
        # self.a_layer = Linear(self.hidden_size, self.emb_size)

        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size) # log2M의 길이를 갖는 따르는 랜덤 가우시안 벡터에 할당하여 인코딩 (평균 0, 분산 I)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True # concat 시 emb_size * 2
        )
        self.out_layer = Linear(self.hidden_size, self.num_q) # 원래 * 2이었으나 축소
        self.dropout_layer = Dropout(self.dropout)


    def forward(self, q, r):
        '''
            q: the questions(KC) sequence with the size of [batch_size, n]
            r: response sequence with the size of [batch_size, n]
        '''

        # Compressive sensing에 의하면 d차원에서의 k-sparse 신호는 모두 원복될 수 있음. (klogd에 변형을 가한 모든 값)
        # 여기서 d차원은 unique exercise이고(M), K-sparse는 원핫인코딩을 거치므로 1-sparse라고 할 수 있음.
        x = self.interaction_emb(q + self.num_q * r) # r텐서를 num_q 만큼 곱해서 확장함
        
        # 여기 BERT 추가해서 돌림
        # BERT, 양 차원 모양 바꾸기 
        # at = self.at_emb_layer(self.bertmodel(input_ids=at_s,
        #                attention_mask=at_t,
        #                token_type_ids=at_m
        #                ).last_hidden_state)
        # at = self.at2_emb_layer(at.permute(0, 2, 1)) # 6, 100, 100 형태로 바꿔줌.

        # v = torch.relu(self.v_emb_layer(torch.concat([x, at], dim=-1)))
        # e = torch.sigmoid(self.e_layer(v))
        # a = torch.tanh(self.a_layer(v))
        
        h, _ = self.lstm_layer(x)
        
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y


def train_model(model, train_loader, valid_loader, num_q, num_epochs, opt, ckpt_path, mode=0, use_wandb=False):
    '''
        Args:
            train_loader: the PyTorch DataLoader instance for training
            test_loader: the PyTorch DataLoader instance for test
            num_epochs: the number of epochs
            opt: the optimization to train this model
            ckpt_path: the path to save this model's parameters
    '''
    max_auc = 0
    loss_means = []  
    aucs = []
    accs = []
    precisions = []
    q_accs = {}
    recalls = []
    f1s = []
    
    wandb_dict = {}
    if use_wandb == True: 
        wandb_dict = {
                    "seed": wandb.config.seed,
                    "dropout": wandb.config.dropout, 
                    "lr": wandb.config.learning_rate,
                    "emb_size": wandb.config.hidden_size,
                    "hidden_size": wandb.config.hidden_size
                }
    
        # "n": 50,
        # "d": 512,
        # "num_attn_heads": 4,
        # "num_tr_layers": 4,
        # "dropout": 0.1
        
    for epoch in range(0, num_epochs):
        time.sleep(35)
        auc_mean = []
        loss_mean = []
        acc_mean = []
        
        model.train()
        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            
            # CSEDM에선 PID_SEQS 대신 LABEL_SEQ로 취급함. 
            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y, t, loss = None, None, None

            if mode == 1:
                # hint_seq 가 다음 정답임
                y, t, loss = dkt_train(model, opt, q, r, qshft_seqs, hint_seqs, num_q, m)
            else:
                y, t, loss = dkt_train(model, opt, q, r, qshft_seqs, rshft_seqs, num_q, m)
            
            common_append(y, t, loss, loss_mean, auc_mean, acc_mean)
            
        loss_mean, auc_mean, acc_mean =  mean_eval(loss_mean, auc_mean, acc_mean)

        log_auc(use_wandb, {
            "epoch": epoch,
            "train_auc": auc_mean,
            "train_acc": acc_mean,
            "train_loss": loss_mean
        })

        if(epoch % 10 == 0):
            print(f"[Train] Epoch: {epoch}, AUC: {auc_mean}, acc: {acc_mean}, Loss Mean: {loss_mean}")

        # Validation
        model.eval()
        with torch.no_grad():
            auc_mean = []
            loss_mean = []
            acc_mean = []
            precision_mean = []
            recall_mean = []
            f1_mean = []
            
            # best_loss = 10 ** 9
            # patience_limit = 3
            # patience_check = 0

            for data in valid_loader:
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
                # CSEDM에선 PID_SEQS 대신 LABEL_SEQ로 취급함. 
                # 현재까지의 입력을 받은 뒤 다음 문제 예측
                y, t, loss = None, None, None
                
                if mode == 1:
                    q, y, t, loss = dkt_test(model, q, r, qshft_seqs, hint_seqs, num_q, m)
                else:
                    q, y, t, loss = dkt_test(model, q, r, qshft_seqs, rshft_seqs, num_q, m)
                                
                # patience_check = early_stopping(best_loss, loss, patience_check)
                # if(patience_check >= patience_limit):
                #     break
                
                try:
                    auc = metrics.roc_auc_score(y_true=t.detach().cpu().numpy(), y_score=y.detach().cpu().numpy())
                except:
                    continue
                
                max_auc = save_auc(model, max_auc, auc, \
                            wandb_dict, # 만약 wandb 체크 안했다면 빈 dict 들어감
                            ckpt_path, use_wandb)
                bin_y = common_append(y, t, loss, loss_mean, auc_mean, acc_mean)
                val_append(t, bin_y, precision_mean, recall_mean, f1_mean)
                q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
                
            loss_mean, auc_mean, acc_mean =  mean_eval(loss_mean, auc_mean, acc_mean)
            precision_mean, recall_mean, f1_mean = mean_eval_ext(precision_mean, recall_mean, f1_mean)

            log_auc(use_wandb, {
                "epoch": epoch,
                "val_auc": auc_mean,
                "val_acc": acc_mean,
                "val_loss": loss_mean
            })
            
            aucs.append(auc_mean)
            loss_means.append(loss_mean)
            accs.append(acc_mean)
            precisions.append(precision_mean)
            recalls.append(recall_mean)
            f1s.append(f1_mean)
            print(f"[Valid] Epoch: {epoch} Result: AUC: {auc_mean}, ACC: {acc_mean}, loss: {loss_mean}")
            
        print(f"========== Finished Epoch: {epoch} ============")
    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s
    
def test_model(model, test_loader, num_q, ckpt_path, mode, use_wandb):
    loss_means = []
    aucs = []
    accs = []
    precisions = []
    q_accs = {}
    recalls = []
    f1s = []
    # Test
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            
            y, t, loss = None, None, None
            
            if mode == 1:
                q, y, t, loss = dkt_test(model, q, r, qshft_seqs, hint_seqs, num_q, m)
            else:
                q, y, t, loss = dkt_test(model, q, r, qshft_seqs, rshft_seqs, num_q, m)
            
            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
            acc = metrics.accuracy_score(t.numpy(), bin_y)
            precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
            recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
            f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
            print(f"[Test] number: {i}, AUC: {auc}, ACC: :{acc} Loss: {loss} ")

            # evaluation metrics
            aucs.append(auc)
            loss_means.append(loss)     
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
        loss_means.append(np.mean(loss_means))
    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s
