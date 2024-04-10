from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftConfig, PeftModel
from model import Router
import torch
from safetensors.torch import load_file
import config

ckpt = 'results/checkpoint-160/model.safetensors'

# 加载配置
# model_config = PeftConfig.from_pretrained(ckpt)

# # 加载模型
# model = PeftModel.from_pretrained(ckpt, config=model_config)

# model.save_pretrained('results/test-ckpt4')

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

new_special_tokens = ['<extra_0>', '<extra_1>', '<extra_2>']
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

# # 检查特殊标记是否已添加
# print(tokenizer.additional_special_tokens)  # 输出: ['<my_special_token>']
# print(tokenizer.all_special_tokens)  # 输出: ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '<mask>', '<my_special_token>']

# 获取新标记的ID
token_id = tokenizer.convert_tokens_to_ids(new_special_tokens)



model = Router()
model.resize_token_embeddings(len(tokenizer))

print('finish loading model')

# model.use_lora()

# torch加载model
model.load_state_dict(load_file(ckpt))
