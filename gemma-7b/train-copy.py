from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import torch
import os
import pandas as pd
from sklearn.metrics import roc_auc_score
import config
from model import Router

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

# Prepare dataset
dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'train.csv')})['data']
# test_dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'test.csv')})['data']
# dataset = dataset.rename_column('Target', 'labels')

dataset = dataset.select(range(1))

# test_dataset = test_dataset.select(range(50))

# print(dataset)

new_special_tokens = ['<extra_0>', '<extra_1>', '<extra_2>']
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

# # 检查特殊标记是否已添加
# print(tokenizer.additional_special_tokens)  # 输出: ['<my_special_token>']
# print(tokenizer.all_special_tokens)  # 输出: ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '<mask>', '<my_special_token>']

# 获取新标记的ID
token_id = tokenizer.convert_tokens_to_ids(new_special_tokens)
# print(token_id)  # 输出: [对应的ID]


def tokenize_function(example):
    prompt = example['Prompt']
    answer = example['Answer']
    src = f'<extra_0>{prompt}<extra_1>{answer}<extra_2>'
    return tokenizer(src, truncation=True)

def to_float(example):
    example['labels'] = torch.tensor(example['Target'], dtype=torch.float32)
    return example

dataset = dataset.map(tokenize_function)
dataset = dataset.map(to_float)

# dataset = dataset.train_test_split(test_size=config.val_rate)

# test_dataset = test_dataset.map(tokenize_function)

print(dataset)

# print(test_dataset)
# exit()

# print(type(dataset['train']['labels'][0]))
# print(dataset['train']['labels'])
# print(dataset['train']['labels'][0])

# train_dataset = dataset['train']
# validation_dataset = dataset['test']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set training arguments

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=30,         # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=0,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    learning_rate=config.learning_rate,
    # load_best_model_at_end=True,  # 在训练结束时加载最佳模型
    # metric_for_best_model="eval_auc-roc",  # 用于比较以找到最佳模型的指标
    evaluation_strategy="no",  # 每个epoch评估一次
    save_strategy="no",  # 每个epoch保存一次
    # fp16=True,  # 使用混合精度训练
    # deepspeed='ds_config.json',
    # save_total_limit=1,
    # load_best_model_at_end=True,
    # save_total_limit=1,
    # resume_from_checkpoint='/home/work/workspace/project/hallucination/results/checkpoint-5845',  # 恢复训练的模型
)


# Define a function to compute metrics
def compute_metrics(eval_pred):
    print('HHH')
    return

    logits, labels = eval_pred
    # If you have a multi-class problem, you might need to apply a softmax to the logits
    # If it's binary classification and your model outputs logits, you might want to use sigmoid
    # Here's an example for binary classification with sigmoid applied to logits:
    # predictions = 1 / (1 + np.exp(-logits)) # applying sigmoid to convert logits to probabilities
    
    # Compute AUC-ROC
    print('logits', logits)
    print('labels', labels)
    auc = roc_auc_score(labels, logits)
    print('calling', auc)
    
    return {"auc-roc": auc}


model = Router()
model.resize_token_embeddings(len(tokenizer))

print('finish loading model')

model.use_lora()

print('finish loading lora')

# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=dataset,         # training dataset
    eval_dataset=dataset,     # evaluation dataset
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# test_set_callback = TestSetPredictionCallback(test_dataset=test_dataset, output_dir=config.pred_dir, trainer=trainer)

# trainer.add_callback(test_set_callback)


# trainer.add_callback(EarlyStoppingCallback(
#     early_stopping_patience=20,    # 如果评估结果在5次评估内不再提升，则停止训练
#     early_stopping_threshold=0.0  # 提升的阈值，如果设置为0，则任何非正的提升都会触发早停
# ))

# Train the model
trainer.train()

# Evaluate the model
trainer.evaluate()

# Save the model
trainer.save_model("my_model")
