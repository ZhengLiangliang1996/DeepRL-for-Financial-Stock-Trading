#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""

from __future__ import print_function
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F



class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Actor1(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Actor1, self).__init__()
        self.state_space = state_space
        self.rnn1 = nn.LSTM(state_space, 64, num_layers=2)
        self.rnn2 = nn.LSTM(64, 32, num_layers=2)
        self.rnn3 = nn.LSTM(32, action_space, num_layers=2)
        # self.fc1  = nn.Linear(32, action_space)


    def forward(self, states):
        states = states.unsqueeze(dim=0)
        lstm_out, _ = self.rnn1(states)
        # print(lstm_out.size())
        lstm_out, _ = self.rnn2(lstm_out)
        # print(lstm_out.size())
        lstm_out, _ = self.rnn3(lstm_out)
        action = lstm_out.squeeze(dim=0)

        return action

class Critic1(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Critic1, self).__init__()
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



class Actor2(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Actor2, self).__init__()
        self.state_space = state_space
        self.rnn1 = nn.LSTM(state_space, 64, num_layers=2)
        self.rnn2 = nn.LSTM(64, 32, num_layers=2)
        self.rnn3 = nn.LSTM(32, action_space, num_layers=2)
        # self.fc1  = nn.Linear(32, action_space)


    def forward(self, states):
        states = states.unsqueeze(dim=0)
        lstm_out, _ = self.rnn1(states)
        # print(lstm_out.size())
        lstm_out, _ = self.rnn2(lstm_out)
        # print(lstm_out.size())
        lstm_out, _ = self.rnn3(lstm_out)
        action = lstm_out.squeeze(dim=0)

        return action

class Critic2(nn.Module):
    def __init__(self, state_space, action_space, hidden_size):
        super(Critic2, self).__init__()
        self.state_space = state_space
        self.rnn1 = nn.LSTM(state_space, 64)
        self.rnn2 = nn.LSTM(64, 32)
        self.rnn3 = nn.LSTM(32, action_space)
        # self.fc1  = nn.Linear(32, action_space)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        states = x.unsqueeze(dim=0)
        lstm_out, _ = self.rnn1(states)
        # print(lstm_out.size())
        lstm_out, _ = self.rnn2(lstm_out)
        # print(lstm_out.size())
        lstm_out, _ = self.rnn3(lstm_out)
        action = lstm_out.squeeze(dim=0)

        return action




