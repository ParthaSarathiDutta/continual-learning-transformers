import torch
from torch import nn
from collections import defaultdict

class EWC:
    def __init__(self, model, dataloader, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.dataloader = dataloader
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in self.params.items():
            self._means[n] = p.clone().detach()

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in self.params.items():
            precision_matrices[n] = torch.zeros_like(p)

        self.model.eval()
        for batch in self.dataloader:
            self.model.zero_grad()
            inputs = {k: v.to(self.device) for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'labels']}
            output = self.model(**inputs)
            loss = output.loss
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    precision_matrices[n] += p.grad.detach() ** 2

        for n in precision_matrices:
            precision_matrices[n] /= len(self.dataloader)

        return precision_matrices

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self._means:
                _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
                loss += _loss.sum()
        return loss
