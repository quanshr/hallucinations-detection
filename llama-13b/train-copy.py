from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os
from sklearn.metrics import roc_auc_score
import config
from model import Router

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True, pad_token='<|endoftext|>')

train_dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'train-train.csv')})['data']
validation_dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'train-valid.csv')})['data']

train_dataset = train_dataset.select(range(1))


new_special_tokens = ['<extra_0>', '<extra_1>', '<extra_2>']
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

token_id = tokenizer.convert_tokens_to_ids(new_special_tokens)

def tokenize_function(example):
    prompt = example['Prompt']
    answer = example['Answer']
    src = f'<extra_0>{prompt}<extra_1>{answer}<extra_2>'
    return tokenizer(src, truncation=True)

def to_float(example):
    example['labels'] = torch.tensor(example['Target'], dtype=torch.float32)
    return example

train_dataset = train_dataset.map(tokenize_function)
train_dataset = train_dataset.map(to_float)
validation_dataset = validation_dataset.map(tokenize_function)
validation_dataset = validation_dataset.map(to_float)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=30,         # total number of training epochs
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    warmup_steps=100,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    learning_rate=config.learning_rate,
    metric_for_best_model="eval_auc-roc",  # 用于比较以找到最佳模型的指标
    evaluation_strategy="no",  # 每个epoch评估一次
    logging_strategy="no",  # 每个epoch打印一次
    save_strategy="no",  # 每个epoch保存一次
    # deepspeed='ds_config.json',
    load_best_model_at_end=True,
    save_total_limit=1,
)

def compute_metrics(eval_pred):
    print('HHH')

    logits, labels = eval_pred
    print('logits', logits)
    print('labels', labels)
    auc = roc_auc_score(labels, logits)
    print('calling', auc)
    
    return {"auc-roc": auc}


model = Router()
model.resize_token_embeddings(len(tokenizer))

print('finish loading model')

model.use_lora()

ckpt = 'results/test-ckpt5'

print('finish loading lora')


# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset,     # evaluation dataset
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

# print('evalution: ', trainer.evaluate())

trainer.train()

print('evalution: ', trainer.evaluate())

trainer.save_model("my_model")
