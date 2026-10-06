"""Trainable typed-history / spatial-scope prototype, not a validated method.

WITNESS references guide a new referent without authorizing edits to the witness.
EDIT references authorize changing a previous selection. No GT pointer, operation
or spatial scope is accepted as an inference input.
"""
import math

import torch
from torch import nn


OPERATIONS = ('new', 'add', 'remove', 'replace', 'refine', 'undo')
ROLES = ('witness', 'edit')


class TypedScopeHead(nn.Module):
    def __init__(self, language_dim, history_dim, pixel_dim=256, hidden_dim=256):
        super().__init__()
        self.query = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, hidden_dim))
        self.history = nn.Sequential(nn.LayerNorm(history_dim), nn.Linear(history_dim, hidden_dim))
        self.role_queries = nn.Linear(hidden_dim, len(ROLES) * hidden_dim)
        self.null_keys = nn.Parameter(torch.zeros(len(ROLES), hidden_dim))
        self.operation = nn.Linear(hidden_dim, len(OPERATIONS))
        self.pixel_key = nn.Conv2d(pixel_dim, hidden_dim, 1)
        self.pixel_query = nn.Linear(hidden_dim, hidden_dim)
        self.scope_bias = nn.Parameter(torch.zeros(()))
        self.hidden_dim = hidden_dim

    def forward(self, query_features, history_features, history_valid, pixel_features):
        if history_valid.shape != history_features.shape[:2]:
            raise ValueError('history validity and history features differ')
        query = self.query(query_features)
        history = self.history(history_features)
        role_queries = self.role_queries(query).reshape(-1, len(ROLES), self.hidden_dim)
        scores = torch.einsum('brd,bnd->brn', role_queries, history) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~history_valid[:, None].bool(), -torch.inf)
        null = (role_queries * self.null_keys[None]).sum(-1, keepdim=True) / math.sqrt(self.hidden_dim)
        # Index zero is a valid null reference, including the first turn.
        pointers = torch.cat([null, scores], dim=-1)
        keys = self.pixel_key(pixel_features)
        scope_logits = torch.einsum('bd,bdhw->bhw', self.pixel_query(query), keys)
        scope_logits = scope_logits / math.sqrt(self.hidden_dim) + self.scope_bias
        return {'operation_logits': self.operation(query), 'role_pointer_logits': pointers,
                'scope_logits': scope_logits, 'scope': scope_logits.sigmoid()}
