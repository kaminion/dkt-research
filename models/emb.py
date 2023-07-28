import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class STFTEmbedding(nn.Module):
    def __init__(self, n_fft, emb_size):
        super(STFTEmbedding, self).__init__()
        self.n_fft = n_fft
        self.hidden_1 = int(n_fft / 2) + 1
        self.hidden_2 = int(n_fft / (n_fft / 4) * 2)
        self.emb_size = emb_size

        self.layer_1 = nn.Sequential(
            nn.Linear(self.hidden_1, self.emb_size),
            nn.Dropout(p=0.3),
            nn.ReLU()
        ) 
        self.layer_2 = nn.Sequential(
            nn.Linear(self.hidden_2, self.emb_size),
            nn.Dropout(p=0.3),
            nn.ReLU()
        ) 

    def forward(self, x):
        x = x.float()
        x_stft = torch.stft(x, self.n_fft, return_complex=True)

        magnitude = torch.abs(x_stft)
        phase = torch.angle(x_stft)
        embedding = torch.cat((magnitude, phase), dim=-1)
        y = self.layer_1(embedding.permute(0, 2, 1))
        y = self.layer_2(y.permute(0, 2, 1))
        
        return y