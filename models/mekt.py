import os
import numpy as np
import torch
from torch import nn
from torch.nn.functional import one_hot, binary_cross_entropy, mse_loss
from sklearn import metrics
"""
MEKT !

"""
class MEKT(nn.Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(MEKT, self).__init__()

        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.parameter_r = nn.Parameter
        
        ##########################################
        # Feature Extraction
        ##########################################
        self.q_emb = nn.Embedding(self.num_q, self.emb_size) # log2M의 길이를 갖는 따르는 랜덤 가우시안 벡터에 할당하여 인코딩 (평균 0, 분산 I)
        self.ua_emb = nn.Embedding(self.num_q * 2, self.emb_size) # Q * R이므로 * 2, question embedding

        self.q_lstm1 = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True)
        self.q_lstm2 = nn.LSTM(2 * self.hidden_size, hidden_size, bidirectional=True)

        self.ua_lstm1 = nn.LSTM(self.emb_size, self.hidden_size, bidirectional=True)
        self.ua_lstm2 = nn.LSTM(2 * self.hidden_size, hidden_size, bidirectional=True)

        self.qlayer_norm = nn.LayerNorm((self.hidden_size * 2, ))
        self.ualayer_norm = nn.LayerNorm((self.hidden_size * 2, ))

        
        ##########################################
        # private encoders
        ##########################################
        self.private_q = nn.Sequential()
        self.private_q.add_module("private_q_1", nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.private_q.add_module("private_q_activation_1", nn.Sigmoid())

        self.private_ua = nn.Sequential()
        self.private_ua.add_module("private_ua_1", nn.Linear(in_features=hidden_size, out_features=hidden_size))
        self.private_ua.add_module("private_ua_activation_1", nn.Sigmoid())


        ##########################################
        # shared encoders
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module("shared_1", nn.Linear(in_features=self.hidden_size, out_features=hidden_size))
        self.shared.add_module("shared_activation_1", nn.Sigmoid())


        ##########################################
        # reconstruct
        ##########################################
        self.recon_q = nn.Sequential()
        self.recon_q.add_module("recon_q_1", nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
        self.recon_ua = nn.Sequential()
        self.recon_ua.add_module("recon_ua_1", nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))

        
        ##########################################
        # shared-private collaborative discriminator
        ##########################################
        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module("sp_discriminator_layer_1", nn.Linear(in_features=self.hidden_size, out_features=3)) # specific 2개, shared 1개


        self.fusion = nn.Sequential()
        self.fusion.add_module("fusion_layer_1", nn.Linear(in_features=self.hidden_size*4, out_features=self.hidden_size*2)) # 특징 * 2 * 2, 특징 개수 * 2
        self.fusion.add_module("fusion_layer_1_dropout", nn.Dropout())
        self.fusion.add_module("fusion_layer_1_activation", nn.ReLU())
        self.fusion.add_module("fusion_layer_final", nn.Linear(in_features=self.hidden_size*2, out_features=1)) # output = num_class

        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)
        

    def extract_feature(self, sequence, lstm1, lstm2, layer_norm):

        b_h1, (h1, _) = lstm1(sequence) # output, ( final_hidden_state 혹은 forward, backward의 hidden state 들을 각각 붙인 것, 마지막은 cell state임 )
        norm_h1 = layer_norm(b_h1)
        _, (h2, _) = lstm2(norm_h1)

        return h1, h2
     

    def encode_sp(self, embed_q, embed_ua):
        
        # private
        self.embed_private_q = self.private_q(embed_q)
        self.embed_private_ua  = self.private_ua(embed_ua)

        # shared
        self.embed_shared_q = self.shared(self.embed_private_q)
        self.embed_shared_ua = self.shared(self.embed_private_ua)
    

    def reconstruct(self):

        self.embed_q = (self.embed_private_q + self.embed_shared_q)
        self.embed_ua = (self.embed_private_ua + self.embed_shared_ua)

        self.embed_q_recon = self.recon_q(self.embed_q)
        self.embed_ua_recon = self.recon_ua(self.embed_ua)
        


    def forward(self, q, r):
        
        batch_size = r.shape[0]
        # 파리미터로 지정되는 텐서
        parameter_r = self.parameter_r(torch.zeros((batch_size, 1))) # response 1개가 빠지므로 하나의 텐서만 지정해주는 것이 바람직하다. 음수일 경우 임베딩 안됨
        ##########################################
        # Feature Extraction
        ##########################################
        emb_q = self.q_emb(q)
        h1_q, h2_q = self.extract_feature(emb_q, self.q_lstm1, self.q_lstm2, self.qlayer_norm)
        extracted_q = torch.cat((h1_q, h2_q))
        # print("1. ", q.shape, r.shape, h1_q.shape, h2_q.shape, extracted_q.shape)
        r = torch.cat([r[:, :-1], parameter_r], dim=-1).long()
        emb_ua = self.ua_emb(q + r)
        h1_ua, h2_ua = self.extract_feature(emb_ua, self.ua_lstm1, self.ua_lstm2, self.ualayer_norm)
        extracted_ua = torch.cat((h1_ua, h2_ua))
        # print("2. ", h1_ua.shape, h2_ua.shape, extracted_ua.shape)

        ##########################################
        # encoding extracted feature and discriminate
        ##########################################
        self.encode_sp(extracted_q, extracted_ua)

        self.dis_private_q = self.sp_discriminator(self.embed_private_q)
        self.dis_private_ua = self.sp_discriminator(self.embed_private_ua)
        self.dis_shared_all = self.sp_discriminator((self.embed_shared_q + self.embed_shared_ua) / 2.0)
        # print("3. ", self.dis_private_q.shape, self.dis_private_ua.shape, self.dis_shared_all.shape)

        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION
        h = torch.cat((self.embed_private_q, self.embed_private_ua, self.embed_shared_q, self.embed_shared_ua), dim=0)
        # print("4. ",self.embed_private_q.shape, self.embed_private_ua.shape, self.embed_shared_q.shape, self.embed_shared_ua.shape, h.shape)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3]), dim=0).permute(1, 0)
        # print("5. ", h.shape)
        y = self.fusion(h)
        # print("predict:", y)
        return torch.sigmoid(y).squeeze() # 차원하나 줄임
    

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

        sim_loss = SIMSE()
        diff_loss = DiffLoss()

        for i in range(0, num_epochs):
            loss_mean = []

            for data in train_loader:
                q, r, _, _, m = data

                self.train()
                
                # 현재까지의 입력을 받은 뒤 다음 문제 예측
                y = self(q.long(), r.long())
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

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                # print(y.shape, m.shape)
                y = torch.masked_select(y, m)
                t = torch.masked_select(r, m)
                # print(f"target: {t}")


                opt.zero_grad()
                loss = binary_cross_entropy(y, t, reduction="mean") # 실제 y^T와 원핫 결합, 다음 answer 간 cross entropy

                ##########################################
                # domain loss
                ##########################################

                ##########################################
                # diff loss
                ##########################################
                loss += diff_loss(self.embed_private_q, self.embed_shared_q)
                loss += diff_loss(self.embed_private_ua, self.embed_shared_ua)
                loss += diff_loss(self.embed_private_q, self.embed_private_ua)


                ##########################################
                # reconstruction loss
                ##########################################
                recon_loss = mse_loss(self.embed_q_recon, self.embed_q) \
                            + mse_loss(self.embed_ua_recon, self.embed_ua)
                recon_loss /= 2.0 # loss 두개를 구하므로..
                loss += recon_loss



                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

            with torch.no_grad():
                for data in test_loader:
                    q, r, _, _, m = data 

                    self.eval()

                    y = self(q.long(), r.long())
                    # y = (y * one_hot(qshift.long(), self.num_q)).sum(-1)

                    # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(r, m).detach().cpu()

                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                    
                    print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean} ")

                    if auc > max_auc : 
                        torch.save(
                            self.state_dict(),
                            os.path.join(
                                ckpt_path, "mekt.ckpt"
                            )
                        )
                        max_auc = auc

                    aucs.append(auc)
                    loss_means.append(loss_mean)

        return aucs, loss_means