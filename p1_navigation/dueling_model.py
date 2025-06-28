import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=128):
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Shared feature layer
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # Value stream
        self.value = nn.Linear(fc2_units, 1)
        # Advantage stream
        self.advantage = nn.Linear(fc2_units, action_size)
        # Dropout
        self.dropout = nn.Dropout(0.25)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        value = self.value(x)
        advantage = self.advantage(x)

        # Combine streams
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q
