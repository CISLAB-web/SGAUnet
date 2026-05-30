"""TA-MoSC MoE from UTANet (AAAI). Vendored from https://github.com/AshleyLuo001/UTANet — pdb breakpoint removed."""

import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple


class Expert(nn.Module):
    def __init__(self, emb_size: int, hidden_rate: int = 2):
        super().__init__()
        hidden_emb = hidden_rate * emb_size
        self.seq = nn.Sequential(
            nn.Conv2d(emb_size, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(hidden_emb, hidden_emb, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_emb),
            nn.ReLU(),
            nn.Conv2d(hidden_emb, emb_size, kernel_size=1, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class MoE(nn.Module):
    def __init__(self, num_experts: int, top: int = 2, emb_size: int = 128, H: int = 224, W: int = 224):
        super().__init__()
        self.experts = nn.ModuleList([Expert(emb_size) for _ in range(num_experts)])
        self.gate1 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate2 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate3 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self.gate4 = nn.Parameter(torch.zeros(emb_size, num_experts), requires_grad=True)
        self._initialize_weights()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.top = top

    def _initialize_weights(self) -> None:
        nn.init.xavier_uniform_(self.gate1)
        nn.init.xavier_uniform_(self.gate2)
        nn.init.xavier_uniform_(self.gate3)
        nn.init.xavier_uniform_(self.gate4)

    def cv_squared(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _process_gate(self, x: torch.Tensor, gate_weights: nn.Parameter) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, emb_size, H, W = x.shape
        x0 = self.gap(x).view(batch_size, emb_size)
        gate_out = F.softmax(x0 @ gate_weights, dim=1)
        expert_usage = gate_out.sum(0)
        top_weights, top_index = torch.topk(gate_out, self.top, dim=1)
        used_experts = torch.unique(top_index)
        unused_experts = set(range(len(self.experts))) - set(used_experts.tolist())
        top_weights = F.softmax(top_weights, dim=1)
        x_expanded = x.unsqueeze(1).expand(batch_size, self.top, emb_size, H, W).reshape(-1, emb_size, H, W)
        y = torch.zeros_like(x_expanded)

        for expert_i, expert_model in enumerate(self.experts):
            expert_mask = (top_index == expert_i).view(-1)
            expert_indices = expert_mask.nonzero().flatten()

            if expert_indices.numel() > 0:
                x_expert = x_expanded[expert_indices]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=expert_indices, source=y_expert)
            elif expert_i in unused_experts and self.training:
                random_sample = torch.randint(0, x.size(0), (1,), device=x.device)
                x_expert = x_expanded[random_sample]
                y_expert = expert_model(x_expert)
                y = y.index_add(dim=0, index=random_sample, source=y_expert)

        top_weights = top_weights.view(-1, 1, 1, 1).expand_as(y)
        y = y * top_weights
        y = y.view(batch_size, self.top, emb_size, H, W).sum(dim=1)

        return y, self.cv_squared(expert_usage)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        y1, loss1 = self._process_gate(x, self.gate1)
        y2, loss2 = self._process_gate(x, self.gate2)
        y3, loss3 = self._process_gate(x, self.gate3)
        y4, loss4 = self._process_gate(x, self.gate4)
        loss = loss1 + loss2 + loss3 + loss4
        return y1, y2, y3, y4, loss
