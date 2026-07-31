from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class CNN1DRegressor(BaseEstimator, RegressorMixin):
    """One-dimensional CNN comparator over the ordered feature vector."""

    def __init__(self, seed: int = 42, epochs: int = 200):
        self.seed = seed
        self.epochs = epochs

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        values = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(1)
        targets = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)

        self.model_ = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(32, 16, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(16, 8, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(8, 4, 3, padding=1),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4 * values.shape[-1], 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.005)
        loss_function = nn.MSELoss()
        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_function(self.model_(values), targets)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            values = torch.tensor(np.asarray(X), dtype=torch.float32).unsqueeze(1)
            return self.model_(values).cpu().numpy().ravel()


class FeatureGraphRegressor(BaseEstimator, RegressorMixin):
    """GNN comparator using a feature graph estimated from training data only."""

    def __init__(self, seed: int = 42, epochs: int = 150, top_k: int = 5):
        self.seed = seed
        self.epochs = epochs
        self.top_k = top_k

    def fit(self, X, y):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        values = np.asarray(X, dtype=float)
        correlations = np.nan_to_num(
            np.abs(np.corrcoef(values, rowvar=False)),
            nan=0.0,
        )
        np.fill_diagonal(correlations, 0.0)

        adjacency = np.zeros_like(correlations)
        neighbor_count = min(self.top_k, max(values.shape[1] - 1, 1))
        for index in range(correlations.shape[0]):
            neighbors = np.argsort(correlations[index])[-neighbor_count:]
            adjacency[index, neighbors] = correlations[index, neighbors]
        adjacency = np.maximum(adjacency, adjacency.T) + np.eye(values.shape[1])
        degree = adjacency.sum(axis=1)
        inverse_sqrt_degree = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-12)))
        normalized = inverse_sqrt_degree @ adjacency @ inverse_sqrt_degree
        self.adjacency_ = torch.tensor(normalized, dtype=torch.float32)

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(1, 32)
                self.layer2 = nn.Linear(32, 16)
                self.output = nn.Linear(16, 1)

            def forward(self, x, adjacency_matrix):
                hidden = torch.relu(self.layer1(x.unsqueeze(-1)))
                hidden = torch.einsum("ij,bjk->bik", adjacency_matrix, hidden)
                hidden = torch.relu(self.layer2(hidden))
                hidden = torch.einsum("ij,bjk->bik", adjacency_matrix, hidden)
                return self.output(hidden.mean(dim=1))

        self.model_ = Network()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.005)
        loss_function = nn.MSELoss()
        train_x = torch.tensor(values, dtype=torch.float32)
        train_y = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)

        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = loss_function(self.model_(train_x, self.adjacency_), train_y)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, X):
        import torch

        self.model_.eval()
        with torch.no_grad():
            values = torch.tensor(np.asarray(X), dtype=torch.float32)
            return self.model_(values, self.adjacency_).cpu().numpy().ravel()
