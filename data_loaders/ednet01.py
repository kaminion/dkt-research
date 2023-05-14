import torch
from torchvision import transforms
# from constant import MAX_STEP, NUM_OF_QUESTIONS  
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import pickle


DATASET_DIR = "./data_loaders/dataset"


# 커스텀 데이터셋은 클래스 상속받아 사용
class EdNet01(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.pickle_path = os.path.join(
            self.dataset_dir, "pickle", "ednet_1"
        )

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        if os.path.exists(os.path.join(self.pickle_path, "q_seqs.pkl")):
            with open(f"{self.pickle_path}/q_seqs.pkl", "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(f"{self.pickle_path}/r_seqs.pkl", "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(f"{self.pickle_path}/u_seqs.pkl", "rb") as f:
                self.u_seqs = pickle.load(f)
            with open(f"{self.pickle_path}/t_seqs.pkl", "rb") as f:
                self.t_seqs = pickle.load(f)


        else: 
            self.q_seqs, self.r_seqs, self.u_seqs, self.t_seqs = self.preprocess()

        self.len = len(self.q_seqs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        print(self.q_seqs[index], self.r_seqs[index], self.t_seqs[index])
        return self.q_seqs[index], self.r_seqs[index], self.t_seqs[index]


    def preprocess(self):
        path = self.pickle_path

        u_seqs = []
        q_seqs = []
        r_seqs = []
        t_seqs = []

        print(os.getcwd())
        file_list = os.listdir("./data_loaders/dataset/KT1_train")

        for file in tqdm(file_list, leave=False):
            u_id = f'{file}'
            u_df = pd.read_csv(f"./data_loaders/dataset/KT1_train/{file}")
            
            u_seqs.append(u_id)
            # 가끔 안되는게 있어서 타입 변경
            u_df['tags'] = u_df['tags'].astype(str)
            q_seqs.append([tags.split(';') for tags in u_df['tags'].values])
            t_seqs.append([u_df['boolean_answer'].values])
            r_seqs.append([u_df['user_answer'].values])
        
        with open(f"{path}/q_seqs.pkl", "wb") as f:
            pickle.dump(q_seqs, f)
        with open(f"{path}/r_seqs.pkl", "wb") as f:
            pickle.dump(r_seqs, f)
        with open(f"{path}/u_seqs.pkl", "wb") as f:
            pickle.dump(u_seqs, f)
        with open(f"{path}/t_seqs.pkl", "wb") as f:
            pickle.dump(t_seqs, f)

        return q_seqs, r_seqs, u_seqs, t_seqs