import os
import torch
from peft import LoraConfig

model_name_or_path = '/home/work/workspace/kaggle2/llama-2-70b-chat-hf'

out_dir = 'results'
os.makedirs(out_dir, exist_ok=True)

rawdata_dir = '../data'
os.makedirs(rawdata_dir, exist_ok=True)

# 7b-lora : 3e-5
learning_rate = 8e-5
val_rate = 0.2
steps_per_eval = 100
max_no_adding_times = 200

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

lora_config = LoraConfig(
            r=8, 
            lora_alpha=16, 
            target_modules="all-linear", # for QLoRA
            lora_dropout=0.05, 
            bias="none",
        )

max_memory_MB = 5000