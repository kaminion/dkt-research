import os 
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler

from torch.nn.functional import one_hot, binary_cross_entropy, mse_loss, softmax
from sklearn import metrics
from sklearn.model_selection import KFold

from models.utils import collate_fn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias=True):
        
        super(ConvLSTMCell, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        # 해당 Conv는 입력값으로 X_(t-1), H_(t-1)을 받음.
        # 출력이 총 4개이고, 각각 hidden state와 shape이 같음.
        self.conv = nn.Conv2d(
            in_channels=self.input_size + self.hidden_size,
            out_channels=4 * self.hidden_size,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
        
    def forward(self, x, cur_state):
        h_cur, c_cur = cur_state
        
        # input Tensor와 hidden state를 채널 별로 이어줌.
        combined = torch.cat([x, h_cur], dim=1)
        
        # 채널 별로 이어준 녀석을 컨볼루션 레이어를 통과시켜준다.
        combined_conv = self.conv(combined)
        
        # combined_conv를 채널 엑시스 기준으로 split해주면 4개로 나눠진다.
        # 여기서 conv filter를 4가지 사용했다고 보면 됨.
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)
        
        # cc_i에 sigmoid 취하면 input_gate
        i = torch.sigmoid(cc_i)
        # cc_f에 sigmoid 취하면 forget_gate
        f = torch.sigmoid(cc_f)
        # cc_o에 sigmoid를 취하면 output_gate
        o = torch.sigmoid(cc_o)
        # cc_g에 tanh를 취하면 input gate와 element-wise하게 곱해질 값이 나타남
        g = torch.sigmoid(cc_g)
        
        # 다음 c state는 forget에 이전 c state를 곱한 것과 input gate 출력에 g를 곱한 것을 더한 것임.
        c_next = f * c_cur + i * g 
        # 다음 hiddent_state는 output gate의 출력에 다음 c state에 tanh를 취한 값을 곱한 것임.
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_size, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        
        # 멀티 레이어를 위한 처리, param수를 레이어 갯수 만큼 맞춰 줌
        self.kernel_size = self._extend_for_multilayer(self.kernel_size, self.num_layers)
        self.hidden_size = self._extend_for_multilayer(self.hidden_size, self.num_layers)
        if not len(self.kernel_size) == len(self.hidden_size) == self.num_layers:
            raise ValueError("Inconsistent list length")
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_size if i == 0 else self.hidden_size[i - 1]
            cell_list.append(
                ConvLSTMCell(cur_input_dim, 
                            self.hidden_size[i], 
                            self.kernel_size[i],
                            self.bias)
            )
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, x, hidden_state=None):
        
        b, _, _, h, w = x.size()
        hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))
        
        layer_output_list = []
        last_state_list = []
        
        seq_len = x.size(1)
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](x=cur_layer_input[:, t, :, :], cur_state=[h, c])
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output
            
            layer_output_list.append(layer_output)
            last_state_list.append([h, c])
            
        
        return layer_output_list, last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        
        return init_states
    
    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
        

class LSTMAE(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(LSTMAE, self).__init__()
        
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout = dropout
        
        ### For Encoding
        self.encoder = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout)
        
        ### For Decoding
        self.decoder = nn.LSTM(self.emb_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)
        
        # self.criterion = nn.MSELoss()

    def forward(self, x):
        _, (hidden_state, cell_state) = self.encoder(x)
        output, (decoder_hidden_state, decoder_cell_state) = self.decoder(x, (hidden_state, cell_state)) # Teacher Forcing
        pred = self.fc(output)
        
        return (hidden_state, cell_state), output


class AUTO(nn.Module):
    def __init__(self, seq_len, num_q, emb_size, hidden_size, num_layers=2, dropout=0.1):
        super(AUTO, self).__init__()


        ##########################################
        # Config Model
        ##########################################
        self.seq_len = seq_len
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
                
        ##########################################
        # Convolutional Auto Encoder
        ##########################################
        self.convEncoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), padding=1, stride=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), padding=0, stride=1),
        )
        self.convDecoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), padding=1, stride=1),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, stride=1),
            nn.Upsample(scale_factor=(2, 2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1), padding=0, stride=1),
        )
        
        
        ##########################################
        # LSTM based AE
        ##########################################
        self.lstmAE = nn.Sequential(
            # nn.Embedding(self.num_q * 2, self.emb_size),
            # nn.LSTM(self.emb_size + 1, self.hidden_size, batch_first=True)
            LSTMAE(self.emb_size, self.hidden_size, self.emb_size, num_layers=self.num_layers, dropout=self.dropout)
        )


        ##########################################
        # Denoising Auto Encoder
        ##########################################        
        self.dEncoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, int(self.hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 4), int(self.hidden_size / 6)),
        )
        self.dDecoder = nn.Sequential(
            nn.Linear(int(self.hidden_size / 6), int(self.hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 4), int(self.hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 2), self.hidden_size)
        )
        
        
        ##########################################
        # Dense Layer for classification
        ##########################################
        self.ffn = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2 + num_layers, int(self.hidden_size / 2)), 
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 2), int(self.hidden_size / 4)),
            nn.ReLU(),
            nn.Linear(int(self.hidden_size / 4), 1),
            nn.Sigmoid()
        )
        
        
        ##########################################
        # criterion
        ##########################################
        self.recon_loss = nn.MSELoss()


    def forward(self, q, r, qshft):
        '''
            q: the questions(KC) sequence with the size of [batch_size, n]
            r: response sequence with the size of [batch_size, n]
        '''
        ##########################################
        # Data Processing
        ##########################################
        x = q + self.num_q * r
        spa_x = torch.stft(x.float(), 198, 1, return_complex=True).float() # 나오는 차원수, 1: ntft / 2 + 1, 2: 입력길이 / hop_length + 1
        
        # 채널 추가 및 사이즈 조정
        cnn_x = spa_x.reshape(spa_x.shape[0], 1, spa_x.shape[1], spa_x.shape[2]).float()
        cnn_x = torchvision.transforms.Resize((100, 100), antialias=True)(cnn_x)

        ##########################################
        # ConvAE
        ##########################################
        hidden_conv = self.convEncoder(cnn_x)
        pred_conv = self.convDecoder(hidden_conv)
                
        ##########################################
        # LSTM-AE
        ##########################################
        (hidden_lstm, cell_lstm), pred_lstm = self.lstmAE(x.float())
        # output, _ = self.lstmAE(spa_x)
        
        # print(hidden_conv.shape, torch.concat([hidden_lstm[0], hidden_lstm[1]], dim=-1).shape)
        
        ##########################################
        # Fusion Heterogenious Feature
        ##########################################
        
        time_fusion = torch.stack([x, pred_lstm], dim=-1)
        freq_fusion = torch.concat([cnn_x, pred_conv], dim=-1)
        freq_fusion = freq_fusion.squeeze(1)
        fusion = torch.concat([time_fusion, freq_fusion], dim=-1)
                                
        ##########################################
        # DAE
        ##########################################
        hidden_d = self.dEncoder(x.float())
        pred_d = self.dDecoder(hidden_d)
                                        
        ##########################################
        # Prediction
        ##########################################
        y = self.ffn(fusion).squeeze(-1)

        return y, cnn_x.float(), pred_conv.float(), x.float(), pred_lstm.float(), None, pred_d.float()
    
def auto_train(model, train_dataset, test_loader, num_q, num_epochs, batch_size, opt, ckpt_path):
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
    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        train_subsampler = SubsetRandomSampler(train_idx)
        val_subsampler = SubsetRandomSampler(val_idx)
        
        # sampler 이용해서 DataLoader 정의
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_subsampler, collate_fn=collate_fn)
        val_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_subsampler, collate_fn=collate_fn)
        
        for i in range(0, num_epochs):
            loss_mean = []
            

            for data in train_loader:
                # q_seqs, r_seqs, qshft_seqs, rshft_seqs, mask_seqs, bert_sentences, bert_sentence_types, bert_sentence_att_mask, proc_atshft_sentences
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data
                model.train()

                # 현재까지의 입력을 받은 뒤 다음 문제 예측
                y, cnn_x, pred_conv, x, pred_lstm, fusion, pred_d = model(q.long(), r.long(), qshft_seqs.long())

                opt.zero_grad()
                y = torch.masked_select(y, m)
                t = torch.masked_select(r, m)

                loss = 0.7 * binary_cross_entropy(y, t) + \
                    0.1 * mse_loss(cnn_x, pred_conv) + \
                    0.1 * mse_loss(x, pred_lstm) + \
                    0.1 * mse_loss(x, pred_d)
                    
                # loss.requires_grad_(True)
                loss.backward()
                opt.step()

                loss_mean.append(loss.detach().cpu().numpy())

        with torch.no_grad():
            for data in val_loader:
                q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

                model.eval()
                
                y, cnn_x, pred_conv, x, pred_lstm, fusion, pred_d = model(q.long(), r.long(), qshft_seqs.long())

                # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
                y = torch.masked_select(y, m).detach().cpu()
                t = torch.masked_select(r, m).detach().cpu()
                
                auc = metrics.roc_auc_score(
                    y_true=t.numpy(), y_score=y.numpy()
                )

                loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
                
                if auc > max_auc : 
                    torch.save(
                        model.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    print(f"Fold:{fold}, previous AUC: {max_auc}, max AUC: {auc}")
                    max_auc = auc

                loss_means.append(loss_mean)

    # 실제 성능측정
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    with torch.no_grad():
        for data in test_loader:
            q, r, qshft_seqs, rshft_seqs, m, bert_s, bert_t, bert_m, q2diff_seqs, pid_seqs, pidshift, hint_seqs = data

            model.eval()
            y, cnn_x, pred_conv, x, pred_lstm, fusion, pred_d = model(q.long(), r.long(), qshft_seqs.long())

            # y와 t 변수에 있는 행렬들에서 마스킹이 true로 된 값들만 불러옴
            y = torch.masked_select(y, m).detach().cpu()
            t = torch.masked_select(r, m).detach().cpu()

            auc = metrics.roc_auc_score(
                y_true=t.numpy(), y_score=y.numpy()
            )

            loss_mean = np.mean(loss_mean) # 실제 로스 평균값을 구함
            
            print(f"Epoch: {i}, AUC: {auc}, Loss Mean: {loss_mean}")

            aucs.append(auc)

    return aucs, loss_means