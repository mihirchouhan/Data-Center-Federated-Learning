"""Compact many-to-one sequence regression model."""
import torch
import torch.nn as nn


class SequenceRegressor(nn.Module):
    def __init__(self, input_size, target_size, hidden_size=32, num_layers=1, model="lstm",
                 node_count=0, node_embedding_size=8, baseline_coefficients=None):
        super().__init__()
        self.node_embedding = nn.Embedding(node_count, node_embedding_size) if node_count else None
        recurrent_size = input_size - 1 + node_embedding_size if node_count else input_size
        recurrent = {"lstm": nn.LSTM, "gru": nn.GRU}[model]
        self.recurrent = recurrent(recurrent_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, target_size)
        self.residual = baseline_coefficients is not None
        if self.residual:
            self.register_buffer("baseline_coefficients", torch.tensor(baseline_coefficients, dtype=torch.float32))
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, inputs):
        baseline = inputs[:, -1, 0:1] if self.residual else None
        if self.node_embedding is not None:
            inputs = torch.cat((inputs[:, :, :-1], self.node_embedding(inputs[:, :, -1].long())), dim=2)
        sequence, _ = self.recurrent(inputs)
        predicted = self.head(sequence[:, -1, :])
        if self.residual:
            predicted = predicted + baseline * self.baseline_coefficients[0] + self.baseline_coefficients[1]
        return predicted
