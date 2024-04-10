from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftConfig, PeftModel
from model import Router
import torch
from safetensors.torch import load_file
import config

ckpt = 'results/checkpoint-160/model.safetensors'

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

new_special_tokens = ['<extra_0>', '<extra_1>', '<extra_2>']
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

token_id = tokenizer.convert_tokens_to_ids(new_special_tokens)

model = Router()
model.resize_token_embeddings(len(tokenizer))

print('finish loading model')

model.load_state_dict(load_file(ckpt))
