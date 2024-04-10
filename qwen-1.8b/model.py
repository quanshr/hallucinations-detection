from transformers import LlamaConfig, LlamaModel
import torch.nn as nn
import torch
from typing import Optional
import config


class Router(nn.Module):

    def __init__(self):
        super().__init__()
        model_config = LlamaConfig.from_pretrained(config.model_name_or_path)  # Llamaconfig works well on Qwen models
        self.model = LlamaModel(model_config)
        self.value_head = nn.Linear(model_config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    
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
        if labels is not None:
            loss = nn.BCELoss()(prob, labels)
        # print('PROB: ', prob)
        # print('LOSS: ', loss)
            return loss, prob
        return prob
