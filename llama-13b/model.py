from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch.nn as nn
import torch
from typing import Optional
import config as cfg
from peft import get_peft_model


class Router(nn.Module):
    # config_class = GemmaConfig

    def __init__(self, config=None):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name_or_path,
            # load_in_4bit=True,
            # device_map='auto',
            # max_memory=max_memory,
            torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            ),
        )
        self.value_head = nn.Linear(5120, 1, dtype=torch.bfloat16)
        self.sigmoid = nn.Sigmoid()
    
    def resize_token_embeddings(self, nums):
        self.model.resize_token_embeddings(nums)

    def forward(self,
                input_ids: torch.LongTensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: torch.FloatTensor = None) -> torch.Tensor:
        outputs = self.model(input_ids,attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        if attention_mask is None:
            last_hidden_states = last_hidden_states[:, -1]
        else:
            last_index = attention_mask.cumsum(dim=1).argmax(dim=1)
            last_hidden_states = last_hidden_states.gather(1, last_index.view(-1, 1, 1).expand(-1, 1, last_hidden_states.size(-1))).squeeze(1)
        prob = self.value_head(last_hidden_states)
        prob = self.sigmoid(prob)
        prob = prob.squeeze(1)
        prob = prob.to(torch.float32)
        print(labels, prob)
        if labels is not None:
            loss = nn.BCELoss()(prob, labels)
            return loss, prob
        return prob


    def use_lora(self):
        self.model = get_peft_model(self.model, cfg.lora_config)
        # print(self.model)