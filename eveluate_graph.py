import os 
import pickle
import numpy as np 
import pandas as pd

model_name = 'akt'

abs_path = f'.{os.path.sep}ckpts{os.path.sep}{model_name}{os.path.sep}'
path = f'ASSIST2009{os.path.sep}'
file_name = 'aucs.pkl'

# TEST DATA SET 에만 수행하는 것이므로..
# ASSISTMENT 2009: 52
epochs = 26

# data = pd.read_csv()
file = None
aucs_np = []


full_path = os.path.join(os.path.join(abs_path, path), file_name)
with open(full_path, "rb") as f:
    file = pickle.load(f)
print(len(file))
div_cnt = int(len(file) / epochs)
# print(len(file), len(file) / epochs)
for i in range(0, div_cnt):
    aucs_np.append(np.array(file[i * epochs:i * epochs + epochs]))
max_values = np.average(file)
std = np.std(file)
print(max_values, std)

    # print(len(file) / 52)

