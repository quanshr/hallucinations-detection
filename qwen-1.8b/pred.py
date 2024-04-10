from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import pandas as pd
import os
import torch
from sklearn.metrics import roc_auc_score
import config
from model import Router

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True, pad_token='<|endoftext|>')

test_dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'test.csv')})['data']

# test_dataset = test_dataset.select(range(200))

dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'train.csv')})['data']


# dataset = dataset.select(range(200))


def tokenize_function(example):
    prompt = example['Prompt']
    answer = example['Answer']
    src = f'<extra_0>{prompt}<extra_1>{answer}<extra_2>'
    return tokenizer(src, truncation=True)

test_dataset = test_dataset.map(tokenize_function)

def to_float(example):
    example['labels'] = torch.tensor(example['Target'], dtype=torch.float32)
    return example

dataset = dataset.map(tokenize_function)
dataset = dataset.map(to_float)


training_args = TrainingArguments(
    output_dir='tmp',
    per_device_eval_batch_size=4,
)

model = Router()

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

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=dataset,
    compute_metrics=compute_metrics,
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer),
)


trainer._load_from_checkpoint('/home/work/workspace/project/hallucination/my_model/checkpoint-16700')


trainer.evaluate()

exit()





predictions = trainer.predict(test_dataset)

predictions = predictions.predictions

# Optionally apply softmax if your model's output is logits
# predictions = softmax(predictions, axis=1)

# Save the predictions to a file
predictions_file = "submission.csv"
df = pd.DataFrame(columns=['Id', 'Target'])

for index, sample in enumerate(test_dataset):
    Id = sample['Id']
    Target = predictions[index]
    df.loc[index] = [Id, Target]

df['Id'] = df['Id'].astype(int)
df.to_csv(predictions_file, index=False)

print(f"Predictions saved in {predictions_file}")
