import torch
from torchvision import transforms
from models.utils import match_seq_len
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import os
import pickle

KAKAO_DATASET_DIR = "/app/input/dataset/dkt-dataset"
DATASET_DIR = f"{KAKAO_DATASET_DIR}/EDNET01/" 
# KAKAO 때문에 추가
SAVE_DIR = "/app/outputs/"
# DATASET_DIR = "datasets/EDNET01/"
Q_SEQ_PICKLE = "q_seqs.pkl"
R_SEQ_PICKLE = "r_seqs.pkl"
U_SEQ_PICKLE = "u_seqs.pkl"
T_SEQ_PICKLE = "t_seqs.pkl"
# U_LIST_PICKLE = "u_list.pkl"
# Q_IDX_PICKLE = "q2idx.pkl"
# Q_DIFF_PICKLE = 'q2diff.pkl'
# P_ID_PICKLE = 'pid.pkl'
# P_LIST_PICKLE = "p_list.pkl"
# HINT_LIST_PICKLE = "hint_use.pkl"

# 커스텀 데이터셋은 클래스 상속받아 사용
class EdNet01(Dataset):
    def __init__(self, seq_len, dataset_dir=DATASET_DIR):
        super().__init__()

        self.dataset_dir = dataset_dir
        self.save_dir = SAVE_DIR

        # 미리 피클에 담겨진 파일들 로딩
        if os.path.exists(os.path.join(self.dataset_dir, Q_SEQ_PICKLE)) & os.path.exists(os.path.join(self.dataset_dir, 'Ednet01.csv')):
            with open(os.path.join(self.dataset_dir, Q_SEQ_PICKLE), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, R_SEQ_PICKLE), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, U_SEQ_PICKLE), "rb") as f:
                self.u_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, T_SEQ_PICKLE), "rb") as f:
                self.t_seqs = pickle.load(f)
        else:
            self.q_seqs, self.r_seqs, self.u_seqs, self.t_seqs = self.preprocess()
                
        self.dataset_path = os.path.join(
            self.dataset_dir, "Ednet01.csv"
        )

        # 유저와 문제 갯수 저장
        self.num_u = self.u_seqs.shape[0]
        self.num_q = self.q_seqs.shape[0]
        
        if seq_len:
            self.q_seqs, self.r_seqs, self.t_seqs, [], [], [] = \
                match_seq_len(self.q_seqs, self.r_seqs, self.t_seqs, [], [], [], seq_len)

        self.len = len(self.q_seqs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        print(self.q_seqs[index], self.r_seqs[index], self.t_seqs[index])
        return self.q_seqs[index], self.r_seqs[index], self.t_seqs[index]


    def preprocess(self):
        u_seqs = []
        qid_seqs = []
        r_seqs = []
        t_seqs = []

        # 새로운 csv 파일 저장을 위한 딕셔너리 생성
        concate_csv = pd.DataFrame()
        
        # 파일 전체적으로 읽기
        file_list = os.listdir(f"{self.dataset_dir}/KT1") # 파일리스트 로드
        # 정답 비교를 위한 파일 추가
        q_df = pd.read_csv(f"{self.dataset_dir}/questions.csv")
        q_df['question_id'] = q_df['question_id'].astype(str)
        q_df.set_index('question_id')
        
        # 파일리스트 로드 후 불러오기 (로딩 적용)
        for file in tqdm(file_list, leave=False):
            u_id = f'{file}'
            u_df = pd.read_csv(f"{self.dataset_dir}/KT1/{file}")
            u_df['correct'] = 0
            # 가끔 안되는게 있어서 타입 변경
            u_df['question_id'] = u_df['question_id'].astype(str)
            
            uids = u_df['question_id']
            qids = q_df['question_id']
            # 각 udf 순회
            for qid in qids:
                print('start ========= ', qid)
                print(u_df.loc[u_df['question_id'] == qid, 'question_id']," === : === ", qid)
                
                cnt = len(u_df.loc[u_df['question_id'] == qid, 'question_id'])
                print(f"count: {cnt} ====================")
                # qid 없다면 넘어감
                if cnt == 0:
                    continue
                
                # 문항번호와 유저번호가 같으면서 정답값도 같다면 1값 할당
                # 1. u_df 문항번호체크
                uc = u_df.loc[u_df['question_id'] == qid, 'user_answer'].values
                print(f"udf: correctness: {uc}")
                
                # 2. q_df 문항번호 체크
                qc = q_df.loc[q_df['question_id'] == qid, 'correct_answer'].values[0]
                print(f"qdf: correctness: {qc}")
                
                
                # 3. boolean 값 비교 후 변경
                u_df.loc[(u_df['question_id'] == qid) & (u_df['user_answer'] == qc), 'correct'] = 1
                print('end ============= ', u_df)
            # 유저 아이디 시퀀스에 넣기
            u_seqs.append(np.array([u_id]))

            qid_seqs.append([u_df['question_id'].values])
            r_seqs.append([u_df['correct'].values])
            t_seqs.append([u_df['user_answer'].values])
            
            concate_csv = pd.concat([concate_csv, u_df])
            
        # 피클에 파일 저장
        with open(os.path.join(self.save_dir, Q_SEQ_PICKLE), "wb") as f:
            pickle.dump(qid_seqs, f)
        with open(os.path.join(self.save_dir, R_SEQ_PICKLE), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.save_dir, U_SEQ_PICKLE), "wb") as f:
            pickle.dump(u_seqs, f)
        with open(os.path.join(self.save_dir, T_SEQ_PICKLE), "wb") as f:
            pickle.dump(t_seqs, f)

        # 새로 만든 파일 저장
        concate_csv.to_csv(os.path.join(self.save_dir, 'Ednet01.csv'), index=False, encoding='utf-8-sig')
        
        return qid_seqs, r_seqs, u_seqs, t_seqs