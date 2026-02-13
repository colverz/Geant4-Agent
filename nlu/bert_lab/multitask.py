from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from nlu.bert_lab.labels import STRUCTURE_LABELS, TOKEN_LABELS

@dataclass
class MultiTaskOutputs:
    structure_logits: torch.Tensor
    token_logits: torch.Tensor
    structure_loss: Optional[torch.Tensor] = None
    token_loss: Optional[torch.Tensor] = None


class MultiTaskBert(nn.Module):
    """
    Multi-task encoder for:
    1) structure classification (sequence-level)
    2) token classification (parameters + entities)

    This is a model skeleton to support future training. It is not wired into
    current training scripts yet.
    """

    def __init__(self, model_name: str, num_structure_labels: int, num_token_labels: int):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        hidden = config.hidden_size

        self.structure_head = nn.Linear(hidden, num_structure_labels)
        self.token_head = nn.Linear(hidden, num_token_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        structure_labels: Optional[torch.Tensor] = None,
        token_labels: Optional[torch.Tensor] = None,
    ) -> MultiTaskOutputs:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        pooled = outputs.last_hidden_state[:, 0, :]
        structure_logits = self.structure_head(pooled)
        token_logits = self.token_head(outputs.last_hidden_state)

        structure_loss = None
        token_loss = None
        if structure_labels is not None:
            structure_loss = nn.CrossEntropyLoss()(structure_logits, structure_labels)
        if token_labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            token_loss = loss_fn(token_logits.view(-1, token_logits.size(-1)), token_labels.view(-1))

        return MultiTaskOutputs(
            structure_logits=structure_logits,
            token_logits=token_logits,
            structure_loss=structure_loss,
            token_loss=token_loss,
        )


def build_multitask_labels() -> Dict[str, int]:
    return {
        "structure_labels": len(STRUCTURE_LABELS),
        "token_labels": len(TOKEN_LABELS),
    }
