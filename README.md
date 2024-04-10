# Detect hallucinations in LLMs

This is the **1st Place Solution** for the Kaggle Competition: [Detecting Hallucinations in LLMs](https://www.kaggle.com/competitions/ml-olympiad-detect-hallucinations-in-llms).

The idea is to concatenate the prompt and response using special tokens and fine-tune a large language model followed by a value head.

Multiple LLMs have been tried as base models, and their optimal results are shown in the table below:

| Model | Fine-tuning Option | AUC-ROC |
| --- | --- | --- |
| Qwen-1_8B-Chat | full-parameter | 0.70545 |
| gemma-2b-it | full-parameter | 0.72737 |
| gemma-7b-it | LoRA | 0.79289 |
| Llama-2-13b-chat | QLoRA | **0.87673** |
| Llama-2-70b-chat | QLoRA | 0.83529 |

The first four models were trained for approximately 10 hours on four Nvidia A100 GPUs, while the Llama-2-70b-chat was trained for about four days on eight Nvidia A100 GPUs, all accelerating with DeepSpeed ZeRO-2.

An interesting finding is that as the model’s parameters increase to a certain extent (under 13B), the optimal performance gradually improves, but when it reaches 70B, the performance actually decreases. This may be due to insufficient data, resulting in larger models not being sufficiently trained. A reasonable speculation is that there will be a corresponding optimal size of model parameters for a specific amount of training data, and both decreasing or increasing parameter sizes will correspondingly reduce the model’s performance on the test set.

You can enter the corresponding model directory and use the following command to fine-tune your own model:

```
deepspeed train.py
```

## Insights for Future Work

The dataset utilizes compiled test instructions with Mistral 7B Instruct to generate responses, including both intrinsic hallucinations (reasoning errors) and extrinsic hallucinations (knowledge errors). As larger models possess stronger reasoning abilities and are trained with more data, the occurrence of hallucinations in both aspects tends to decrease accordingly. So a feasible approach is to employ more powerful models, such as GPT-4, to examine the model responses, or to directly answer the questions, and use the responses as references added as a part of input for the classifier. Additionally, to detect extrinsic hallucinations, leveraging an external knowledge database for retrieval augmentation and utilizing GPT-4 for analysis and integration, or direct generation, is also a promising method.