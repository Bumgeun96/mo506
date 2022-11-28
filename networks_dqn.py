import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Agent, self).__init__()
        self.hidden_dim = 128
        self.input_linear = torch.nn.Linear(num_inputs, self.hidden_dim)
        self.middle_linear1 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.middle_linear2 = torch.nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_linear = torch.nn.Linear(self.hidden_dim, num_outputs)
        self.Leakyrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.input_linear(x).clamp(min=0))
        x = self.relu(self.middle_linear1(x).clamp(min=0))
        x = self.relu(self.middle_linear2(x).clamp(min=0))
        q_pred = self.output_linear(x)
        return q_pred

    def save_model(self, net, filename):
        torch.save(net.state_dict(), filename)