from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel

class ConcisenessClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        category_cardinalities: Dict[str, int],
        num_numeric_features: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.cat_embeddings = nn.ModuleDict()
        cat_total_dim = 0
        for name, size in category_cardinalities.items():
            emb_dim = min(32, max(4, (size + 1) // 2))
            self.cat_embeddings[name] = nn.Embedding(size, emb_dim)
            cat_total_dim += emb_dim

        self.numeric_proj = nn.Sequential(
            nn.Linear(num_numeric_features, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        combined_dim = hidden_size + cat_total_dim + 16
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        categorical_inputs: Dict[str, torch.Tensor],
        numeric_features: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            text_repr = outputs.pooler_output
        else:
            text_repr = outputs.last_hidden_state[:, 0, :]

        cat_parts = [self.cat_embeddings[name](categorical_inputs[name]) for name in self.cat_embeddings]
        cat_repr = torch.cat(cat_parts, dim=-1) if cat_parts else torch.empty_like(text_repr[:, :0])
        num_repr = self.numeric_proj(numeric_features)
        fused = torch.cat([text_repr, cat_repr, num_repr], dim=-1)
        logits = self.classifier(fused).squeeze(-1)
        return logits
