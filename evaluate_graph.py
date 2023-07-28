import os 
import pickle
import numpy as np 
import pandas as pd

<<<<<<< HEAD:eveluate_graph.py
model_name = 'akt'
=======
model_name = 'dkvmn+'
>>>>>>> ea95c94167ca1e37d8e226f2821d2aa68f9dd2b4:evaluate_graph.py

abs_path = f'.{os.path.sep}ckpts{os.path.sep}{model_name}{os.path.sep}'
path = f'ASSIST2009{os.path.sep}'
file_name = 'aucs.pkl'

# TEST DATA SET 에만 수행하는 것이므로..
# ASSISTMENT 2009: 52, ASSISTMENT 2012: 379
epochs = 52

# data = pd.read_csv()
file = None
aucs_np = []


full_path = os.path.join(os.path.join(abs_path, path), file_name)
with open(full_path, "rb") as f:
    file = pickle.load(f)
<<<<<<< HEAD:eveluate_graph.py
print(len(file))
div_cnt = int(len(file) / epochs)
# print(len(file), len(file) / epochs)
for i in range(0, div_cnt):
    aucs_np.append(np.array(file[i * epochs:i * epochs + epochs]))
max_values = np.average(file)
=======
# print(len(file))
# div_cnt = int(len(file) / epochs)
# for i in range(0, div_cnt):
#     aucs_np.append(file[i * epochs:i * epochs + epochs])
med_values = np.median(file)
max_values = np.max(file)
# print(len(max_values), max_values, maximum_AUC)
>>>>>>> ea95c94167ca1e37d8e226f2821d2aa68f9dd2b4:evaluate_graph.py
std = np.std(file)
print(med_values, max_values, std)

    # print(len(file) / 52)

