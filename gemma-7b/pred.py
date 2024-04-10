from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import pandas as pd
import torch
import os
import config
from sklearn.metrics import roc_auc_score
from model import Router
from safetensors.torch import load_file


tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)
new_special_tokens = ['<extra_0>', '<extra_1>', '<extra_2>']
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})
token_id = tokenizer.convert_tokens_to_ids(new_special_tokens)

model = Router()
model.resize_token_embeddings(len(tokenizer))

print('finish loading model')

model.use_lora()


ckpt = 'results/checkpoint-13352/model.safetensors'
# safetensors加载model
model.load_state_dict(load_file(ckpt))


test_dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'test.csv')})['data']
# test_dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'train.csv')})['data']


# test_dataset = test_dataset.select(range(200))

def tokenize_function(example):
    prompt = example['Prompt']
    answer = example['Answer']
    src = f'<extra_0>{prompt}<extra_1>{answer}<extra_2>'
    return tokenizer(src, truncation=True)

test_dataset = test_dataset.map(tokenize_function)

training_args = TrainingArguments(
    output_dir='tmp',
    per_device_eval_batch_size=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer),
)

predictions = trainer.predict(test_dataset)

predictions = predictions.predictions

# Optionally apply softmax if your model's output is logits
# predictions = softmax(predictions, axis=1)

# Save the predictions to a file

if 'Target' in test_dataset[0].keys():
    predictions_file = "train.csv"
    df = pd.DataFrame(columns=['Id', 'Label', 'Target'])

    for index, sample in enumerate(test_dataset):
        Id = sample['Id']
        Label = sample['Target']
        Target = predictions[index]
        df.loc[index] = [Id, Label, Target]

    df['Id'] = df['Id'].astype(int)
    df['Label'] = df['Label'].astype(int)
    df.to_csv(predictions_file, index=False)

    labels = df['Label']
    logits = df['Target']
    auc = roc_auc_score(labels, logits)

    print(f"Predictions saved in {predictions_file}")
    print(f"AUC: {auc}")

else:
    predictions_file = "submission.csv"
    df = pd.DataFrame(columns=['Id', 'Target'])

    for index, sample in enumerate(test_dataset):
        Id = sample['Id']
        Target = predictions[index]
        df.loc[index] = [Id, Target]

    df['Id'] = df['Id'].astype(int)
    df.to_csv(predictions_file, index=False)

    print(f"Predictions saved in {predictions_file}")
