import torch 
from torch import nn
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, MultiheadAttention, LayerNorm


class NCF(nn.Module):
    def __init__(self, layers, num_users, num_items, latent_dim_mf, latent_dim_mlp, dropout=0.2):
        super(NCF, self).__init__()
        self.num_users = num_users # 유저 수 
        self.num_items = num_items # 문제 수
        self.dropout = dropout
        
        self.latent_dim_mf = latent_dim_mf
        self.latent_dim_mlp = latent_dim_mlp
        
        self.layers = layers
        
        self.emb_user_mlp = Embedding(self.num_users, self.latent_dim_mlp)
        self.emb_item_mlp = Embedding(self.num_items, self.latent_dim_mlp)
        self.emb_user_mf = Embedding(self.num_users, self.latent_dim_mf)
        self.emb_item_mf = Embedding(self.num_items, self.latent_dim_mf)
        
        self.fc_layers = nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
        
        self.affine_output = nn.Linear() 
        self.logistic = nn.Sigmoid()
        

    def forward(self, u, i): # q / r
        '''
            q: the questions(KC) sequence with the size of [batch_size, n]
            r: response sequence with the size of [batch_size, n]
        '''

        # Compressive sensing에 의하면 d차원에서의 k-sparse 신호는 모두 원복될 수 있음. (klogd에 변형을 가한 모든 값)
        # 여기서 d차원은 unique exercise이고(M), K-sparse는 원핫인코딩을 거치므로 1-sparse라고 할 수 있음.
        
        user_emb_mlp = self.emb_user_mlp(u)
        item_emb_mlp = self.emb_item_mlp(i)
        user_emb_mf = self.emb_user_mf(u)
        item_emb_mf = self.emb_item_mf(i)
        
        mlp_vector = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mf_vector = torch.mul(user_emb_mf, item_emb_mf)
        
        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)
        
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        y = self.logistic(logits)

        return y