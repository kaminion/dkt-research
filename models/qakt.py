import os 
import numpy as np
import torch
from torch import nn
from torch.nn import Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy, mse_loss
from sklearn import metrics
from transformers import BertModel, BertConfig
from models.utils import bert_tokenizer


class QAKT(nn.Module):
    def __init__(self, num_q, emb_size, hidden_size) :
        super(QAKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.interaction_emb = nn.Embedding(2 * self.num_q, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, 1)
        self.dropout_layer = Dropout()

        # for answer text processing
        bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
        self.size_layer = nn.Sequential()
        self.size_layer.add_module("size_layer", nn.Linear(in_features=768, out_features=self.hidden_size))
        self.size_layer.add_module("size_layer_activation", nn.ReLU())
        self.size_layer.add_module("size_layer_norm", nn.LayerNorm(self.hidden_size))

        self.recon = nn.Sequential()
        self.recon.add_module("recon_layer", nn.Linear(self.hidden_size, self.hidden_size))
        self.recon.add_module("discriminator", nn.Linear(self.hidden_size, 1)) # text generator


    def forward(self, q, r, bert_sent, bert_sent_type, bert_sent_mask):
        x = q + self.num_q * r 
        x = self.interaction_emb(x)
        h, _ = self.lstm_layer(x)
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y).squeeze()
        # 여기까지 일반 DKT
        # reconstruct y
        hb = self.bertmodel(input_ids=bert_sent,
                       attention_mask=bert_sent_mask,
                       token_type_ids=bert_sent_type
                       )
        # print(type(hb.hidden_states))
        sb = self.size_layer(hb.last_hidden_state)
        ry = self.recon(sb).squeeze()
        
        return y, ry
    
    def train_model(self, train_loader, test_loader, num_epochs, opt, ckpt_path):
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
                q, r, _, rshft, m, bert_sentences, bert_sentence_types, bert_sentence_att_mask, atshft_sentence = data
                self.train()
                
                # 현재까지의 입력을 받은 뒤 다음 문제 예측
                y, ry = self(q.long(), r.long(), bert_sentences, bert_sentence_types, bert_sentence_att_mask)
                # one-hot은 sparse한 벡터를 아웃풋으로 가짐, 주어진 텐서의 값을 인덱스 번호로 보고 원 핫 임베딩을 함. 뒤에는 num_classes.
                # 예를 들어 tensor([0, 1, 2, 0, 1]) 인 경우, 인덱스 번호를 원핫으로 보고 
                # tensor([[1, 0, 0, 0, 0],
                # [0, 1, 0, 0, 0],
                # [0, 0, 1, 0, 0],
                # [1, 0, 0, 0, 0],
                # [0, 1, 0, 0, 0]]) 로 배출함
                # 여기서는 해당 문제임을 알기 위해 원 핫 사용
                # print(f"y: {y}")
                # y = (y * one_hot(qshift.long(), self.num_q)).sum(-1) # 이 과정이 원 핫, 곱해서 정답과 매핑시킴 
                # print(f"{bert_tokenizer.decode(ry[0][1], skip_special_tokens=True)}, {bert_tokenizer.decode(ry[1][1], skip_special_tokens=True)}, \
                #       {bert_tokenizer.decode(ry[2][1], skip_special_tokens=True)}, {bert_tokenizer.decode(ry[3][1], skip_special_tokens=True)}, {bert_tokenizer.decode(ry[4][1], skip_special_tokens=True)}")

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m)
                t = torch.masked_select(rshft, m)
                # print(f"target: {t}")


                opt.zero_grad()
                loss = binary_cross_entropy(y, t, reduction="mean") # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy

                ##########################################
                # reconstruction loss
                ##########################################
                recon_loss = mse_loss(ry.float(), atshft_sentence.float())
                loss += recon_loss

                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, _, rshft, m, bert_sentences, bert_sentence_types, bert_sentence_att_mask, qshft_seqs = data 

                    self.eval()

                    y, ry = self(q.long(), r.long(), bert_sentences, bert_sentence_types, bert_sentence_att_mask)
                    # y = (y * one_hot(qshift.long(), self.num_q)).sum(-1)

                    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(rshft, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                    
                    print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean} {bert_tokenizer.decode(ry[0])}, {bert_tokenizer.decode(qshft_seqs[0])} ")

                    if auc > max_auc : 
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "qakt.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means