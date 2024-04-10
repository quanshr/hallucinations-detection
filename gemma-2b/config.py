import os
import torch

model_name_or_path = '/home/work/workspace/envs/gemma-2b-it'

out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

rawdata_dir = '../data'
os.makedirs(rawdata_dir, exist_ok=True)

pred_dir = 'predictions'
os.makedirs(pred_dir, exist_ok=True)

val_rate = 0.2
lr = 4e-8
steps_per_eval = 100
max_no_adding_times = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)