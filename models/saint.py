import os 
import numpy as np
import torch 

from tqdm import tqdm 

from torch.nn import Module, Parameter, Embedding, Linear, Transformer
from torch.nn.init import normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics 
from models.utils import cal_acc_class
from transformers import BertModel, BertConfig


class SAINT(Module):
    def __init__(
            self, num_q, n, d, num_attn_heads, dropout, num_tr_layers=1
    ):
        super(SAINT, self).__init__()
        self.num_q = num_q
        self.n = n
        self.d = d
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_tr_layers = num_tr_layers

        self.E = Embedding(self.num_q, self.d)
        self.R = Embedding(2, self.d)
        self.P = Parameter(torch.Tensor(self.n, self.d))
        self.S = Parameter(torch.Tensor(1, self.d))

        normal_(self.P)
        normal_(self.S)

        self.transformer = Transformer(
            self.d,
            self.num_attn_heads,
            num_encoder_layers=self.num_tr_layers,
            num_decoder_layers=self.num_tr_layers,
            dropout=self.dropout
        )
        
        # BERT for feature extraction
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        self.at_emb_layer = Linear(768, self.d)
        self.at2_emb_layer = Linear(512, 100)
        self.v_emb_layer = Linear(self.d * 2, self.d)
        
        self.e_layer = Linear(self.d, self.d)
        self.a_layer = Linear(self.d, self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, at_s, at_t, at_m):
        batch_size = r.shape[0]

        E = self.E(q).permute(1, 0, 2)
        R = self.R(r[:, :-1]).permute(1, 0, 2)
        S = self.S.repeat(batch_size, 1).unsqueeze(0)
        R = torch.cat([S, R], dim=0)

        # BERT, 양 차원 모양 바꾸기 
        # A = self.at_emb_layer(self.bertmodel(input_ids=at_s,
        #                attention_mask=at_t,
        #                token_type_ids=at_m
        #                ).last_hidden_state)
        # A = self.at2_emb_layer(A.permute(0, 2, 1)) # 어텐션에 들어가는 형식으로 바까줌

        P = self.P.unsqueeze(1)

        mask = self.transformer.generate_square_subsequent_mask(self.n).cuda()
        R = self.transformer(
            E + P, R + P, mask, mask, mask
        )
        R = R.permute(1, 0, 2)

        p = torch.sigmoid(self.pred(R)).squeeze(-1)

        return p
    
    
def train_model(model, train_loader, valid_loader, test_loader, num_q, num_epochs, batch_size, opt, ckpt_path):
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
    accs = []
    q_accs = {}
    precisions = []
    recalls = []
    f1s = []
    
    max_auc = 0

    for i in tqdm(range(0, num_epochs)):
        loss_mean = []

        for data in train_loader:
            # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
            
            # CSEDM에선 PID_SEQS 대신 LABEL_SEQ로 취급함. 
            
            
            model.train()
            # 현재까지의 입력을 받은 뒤 다음 문제 예측
            y = model(q.long(), r.long(), bert_s, bert_t, bert_m)

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            y = torch.masked_select(y, m)
            t = torch.masked_select(pid_seqs, m) # CSEDM은 r 대신 pid_seqs

            opt.zero_grad()
            loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy
            loss.backward()
            opt.step()
            
            loss_mean.append(loss.detach().cpu().numpy())
        auc = metrics.roc_auc_score(
            y_true=t.detach().cpu().numpy(), y_score=y.detach().cpu().numpy()
        )
        bin_y = [1 if p >= 0.5 else 0 for p in y.detach().cpu().numpy()]
        acc = metrics.accuracy_score(t.detach().cpu().numpy(), bin_y)

        print(f"[Train] number: {i}, AUC: {auc}, ACC: {acc} ")
        # Validation
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()

                y = model(q.long(), r.long(), bert_s, bert_t, bert_m)

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(pid_seqs, m).detach().cpu() # CSEDM은 r 대신 pid_seqs

                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )
                bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
                acc = metrics.accuracy_score(t.numpy(), bin_y)
                loss = binary_cross_entropy(y, t)
                
                print(f"[Valid] number: {i}, AUC: {auc}, ACC: {acc} Loss: {loss} ")

                if auc > max_auc : 
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    max_auc = auc
    # Test
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

            model.eval()

            y = model(q.long(), r.long(), bert_s, bert_t, bert_m)

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            q = torch.masked_select(q, m).detach().cpu()
            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(pid_seqs, m).detach().cpu() # CSEDM은 r 대신 pid_seqs

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )
            bin_y = [1 if p >= 0.5 else 0 for p in y.numpy()]
            acc = metrics.accuracy_score(t.numpy(), bin_y)
            precision = metrics.precision_score(t.numpy(), bin_y, average='binary')
            recall = metrics.recall_score(t.numpy(), bin_y, average='binary')
            f1 = metrics.f1_score(t.numpy(), bin_y, average='binary')
            
            loss = binary_cross_entropy(y, t) # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy

            print(f"[Test] number: {i}, AUC: {auc}, ACC: :{acc} Loss: {loss} ")

            # evaluation metrics
            aucs.append(auc)
            loss_mean.append(loss)     
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
            
            q_accs, cnt = cal_acc_class(q.long(), t.long(), bin_y)
        loss_means.append(np.mean(loss_mean))


    return aucs, loss_means, accs, q_accs, cnt, precisions, recalls, f1s