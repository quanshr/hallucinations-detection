from transformers import GemmaConfig, GemmaModel, PreTrainedModel
import torch.nn as nn
import torch
from typing import Optional
import config as cfg


class Router(PreTrainedModel):
    config_class = GemmaConfig

    def __init__(self, config=None):
        model_config = GemmaConfig.from_pretrained(cfg.model_name_or_path)
        super().__init__(config=model_config)
        self.model = GemmaModel(model_config)
        self.value_head = nn.Linear(model_config.hidden_size, 1)
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
        if labels is not None:
            loss = nn.BCELoss()(prob, labels)
            return loss, prob
        return prob
