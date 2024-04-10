from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from datasets import load_dataset
import torch
import os
from sklearn.metrics import roc_auc_score
import config
from model import Router

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True, pad_token='<|endoftext|>')

# Prepare dataset
dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'train.csv')})['data']
# dataset = dataset.rename_column('Target', 'labels')

# dataset = dataset.select(range(100))

# print(dataset)

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

dataset = dataset.train_test_split(test_size=config.val_rate)

print(dataset)
# exit()

# print(type(dataset['train']['labels'][0]))
# print(dataset['train']['labels'])
# print(dataset['train']['labels'][0])

train_dataset = dataset['train']
validation_dataset = dataset['test']

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Set training arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=20,         # total number of training epochs
    per_device_train_batch_size=4,  # batch size per device during training
    per_device_eval_batch_size=4,   # batch size for evaluation
    warmup_steps=100,                 # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    learning_rate=5e-8,
    # load_best_model_at_end=True,  # 在训练结束时加载最佳模型
    # metric_for_best_model="eval_auc-roc",  # 用于比较以找到最佳模型的指标
    evaluation_strategy="epoch",  # 每个epoch评估一次
    save_strategy="epoch",  # 每个epoch保存一次
    # fp16=True,  # 使用混合精度训练
    # deepspeed='ds_config.json',
    # save_total_limit=1,
)


# Define a function to compute metrics
def compute_metrics(eval_pred):
    print('HHH')

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



# Initialize Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=validation_dataset,     # evaluation dataset
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

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
