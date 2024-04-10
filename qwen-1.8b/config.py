import os
import torch

model_name_or_path = '/home/work/workspace/envs/Qwen1_8'

out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

rawdata_dir = '../data'

val_rate = 0.2
lr = 4e-8
steps_per_eval = 100
max_no_adding_times = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)