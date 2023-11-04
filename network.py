import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space, reward_size):
        super(Actor, self).__init__()
        self.action_space = action_space
        self.reward_size = reward_size
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs + reward_size, 2*hidden_size)
        nn.init.normal_(self.linear1.weight, 0.0, 0.02)
        self.linear2 = nn.Linear(2*hidden_size, hidden_size)
        nn.init.normal_(self.linear2.weight, 0.0, 0.02)

        self.mu = nn.Linear(hidden_size, num_outputs)
        torch.nn.init.uniform_(self.mu.weight, a=-3e-3, b=3e-3)

    def forward(self, inputs, preference):
        x = inputs
        x = torch.cat((x, preference), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        mu = torch.tanh(self.mu(x))
        return mu


class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space, reward_size):
        super(Critic, self).__init__()
        self.action_space = action_space
        self.reward_size = reward_size
        num_outputs = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs + num_outputs + reward_size, 2*hidden_size)
        nn.init.normal_(self.linear1.weight, 0.0, 0.02)
        self.ln1 = nn.LayerNorm(2*hidden_size)

        self.linear2 = nn.Linear(2*hidden_size, hidden_size)
        nn.init.normal_(self.linear2.weight, 0.0, 0.02)
        self.ln2 = nn.LayerNorm(hidden_size)

        self.V = nn.Linear(hidden_size, reward_size)
        torch.nn.init.uniform_(self.V.weight, a=-3e-3, b=3e-3)

    def forward(self, inputs, actions, preference):
        x = torch.cat((inputs, actions), dim=1)
        x = torch.cat((x, preference), dim=1)
        x = self.linear1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)
        V = self.V(x)
        return V