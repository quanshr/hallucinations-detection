from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
import pandas as pd
import os
import config
from model import Router

tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, trust_remote_code=True)

model = Router.from_pretrained("results/checkpoint-20", ignore_mismatched_sizes=True)

test_dataset = load_dataset('csv', data_files={'data': os.path.join(config.rawdata_dir, 'test.csv')})['data']

test_dataset = test_dataset.select(range(200))

new_special_tokens = ['<extra_0>', '<extra_1>', '<extra_2>']
tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

def tokenize_function(example):
    prompt = example['Prompt']
    answer = example['Answer']
    src = f'<extra_0>{prompt}<extra_1>{answer}<extra_2>'
    return tokenizer(src, truncation=True)

test_dataset = test_dataset.map(tokenize_function)

training_args = TrainingArguments(
    output_dir='tmp',
    per_device_eval_batch_size=4,
)

model.resize_token_embeddings(len(tokenizer))

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
predictions_file = "submission.csv"
df = pd.DataFrame(columns=['Id', 'Target'])

for index, sample in enumerate(test_dataset):
    Id = sample['Id']
    Target = predictions[index]
    df.loc[index] = [Id, Target]

df['Id'] = df['Id'].astype(int)
df.to_csv(predictions_file, index=False)

print(f"Predictions saved in {predictions_file}")