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


# configuration of device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# base line: 3-layer of linear layer with RELU activataion
class DQN(nn.ModuleDict):
    def __init__(self, state_space, action_space, hidden_fc1 = 128, hidden_fc2 = 64):
        """
        Baseline for the LunarLander senario network architecturec
        3 linear layers with relu activation
        """
        super(DQN, self).__init__()

        self.state_space = state_space
        self.action_space = action_space
        self.hidden_fc1 = hidden_fc1
        self.hidden_fc2 = hidden_fc2

        self.layer1 = nn.Linear(state_space, hidden_fc1)
        self.layer2 = nn.Linear(hidden_fc1, hidden_fc2)
        self.output = nn.Linear(hidden_fc2, action_space)

    def forward(self, states):
        """
        Map the state to action value
        """
        x = self.layer1(states)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        action = self.output(x)

        return action

class DQN_1(nn.Module):
    def __init__(self, state_space, action_space):
        """ Classical DQN 3-layer CNN
        """
        super(DQN_1, self).__init__()
        # input channel: 1, length is 6
        self.state_space = state_space
        self.action_space = action_space
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 2, kernel_size = 2,
                               dilation = 1, bias = True)
        self.bn1   = nn.BatchNorm1d(2)
        self.dp1   = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels = 2, out_channels = 4, kernel_size = 2,
                               dilation = 2, bias = True)
        self.bn2   = nn.BatchNorm1d(4)
        self.dp2   = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size = 2,
                               dilation = 4, bias = True)
        self.bn3   = nn.BatchNorm1d(8)
        self.dp3   = nn.Dropout(0.25)

        self.conv4 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 2,
                               dilation = 8, bias = True)
        self.bn4   = nn.BatchNorm1d(16)
        self.dp4   = nn.Dropout(0.25)

        self.conv5 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 2,
                               dilation = 16, bias = True)
        self.bn5   = nn.BatchNorm1d(32)
        self.dp5   = nn.Dropout(0.25)

        # 35 * 59
        self.dense1= nn.Linear(1888, 1024)
        self.dp6   = nn.Dropout(0.25)
        self.dense2= nn.Linear(1024, 512)
        self.dp7   = nn.Dropout(0.25)
        self.dense3= nn.Linear(512, self.action_space)
        self.dp8   = nn.Dropout(0.25)

    def forward(self, states):
        #x = states.view(-1, 1, self.state_space)
        x = states.unsqueeze(dim=1)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dp1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dp2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dp3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.dp4(x)

        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = self.dp5(x)

        x = x.view(-1, 32 * 59)
        x = F.selu(self.dense1(x))
        x = self.dp6(x)
        x = F.selu(self.dense2(x))
        x = self.dp7(x)
        x = F.selu(self.dense3(x))
        action = self.dp8(x)
        action = F.softmax(action)
        return action

# Deuling Double DQN
class DQN_2(nn.Module):
    def __init__(self, state_space, action_space):
        """ Classical DQN 3-layer CNN
        """
        super(DQN_2, self).__init__()
        # input channel: 1, length is 6
        self.state_space = state_space
        self.action_space = action_space
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 2, kernel_size = 2,
                               dilation = 1, bias = True)
        self.bn1   = nn.BatchNorm1d(2)
        self.dp1   = nn.Dropout(0.25)

        self.conv2 = nn.Conv1d(in_channels = 2, out_channels = 4, kernel_size = 2,
                               dilation = 2, bias = True)
        self.bn2   = nn.BatchNorm1d(4)
        self.dp2   = nn.Dropout(0.25)

        self.conv3 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size = 2,
                               dilation = 4, bias = True)
        self.bn3   = nn.BatchNorm1d(8)
        self.dp3   = nn.Dropout(0.25)

        self.conv4 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 2,
                               dilation = 8, bias = True)
        self.bn4   = nn.BatchNorm1d(16)
        self.dp4   = nn.Dropout(0.25)

        self.conv5 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 2,
                               dilation = 16, bias = True)
        self.bn5   = nn.BatchNorm1d(32)
        self.dp5   = nn.Dropout(0.25)

        # 35 * 59 state_value
        self.dense1= nn.Linear(1888, 1024)
        self.dp6   = nn.Dropout(0.25)
        self.dense2= nn.Linear(1024, 512)
        self.dp7   = nn.Dropout(0.25)
        self.dense3= nn.Linear(512, 1)
        self.dp8   = nn.Dropout(0.25)

        # advantage value
        self.dense4 = nn.Linear(1888, 1024)
        self.dp9    = nn.Dropout(0.25)
        self.dense5 = nn.Linear(1024, 512)
        self.dp10   = nn.Dropout(0.25)
        self.dense6 = nn.Linear(512, self.action_space)
        self.dp11   = nn.Dropout(0.25)



    def forward(self, states):
        x = states.view(-1, 1, self.state_space)
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.dp1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.dp2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.dp3(x)

        x = F.relu(self.conv4(x))
        x = self.bn4(x)
        x = self.dp4(x)

        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x1 = self.dp5(x)

        x = x1.view(-1, 32 * 59)
        x = F.selu(self.dense1(x))
        x = self.dp6(x)
        x = F.selu(self.dense2(x))
        x = self.dp7(x)
        x = F.selu(self.dense3(x))
        state_value = self.dp8(x)

        x = x1.view(-1, 32 * 59)
        x = F.selu(self.dense4(x))
        x = self.dp9(x)
        x = F.selu(self.dense5(x))
        x = self.dp10(x)
        x = F.selu(self.dense6(x))
        advantage_value = self.dp11(x)

        advantage_mean = torch.Tensor.mean(advantage_value, dim=1, keepdim=True)
        action = state_value.expand([-1, 3]) + (advantage_value - advantage_mean.expand([-1, 3]))
        return action


# class Actor(nn.Module):

    # def __init__(self,input_size,action_size,256):
        # super(Actor, self).__init__()
        # self.fc1 = nn.Linear(input_size,hidden_size)
        # self.fc2 = nn.Linear(hidden_size,hidden_size)
        # self.fc3 = nn.Linear(hidden_size,action_size)

    # def forward(self,x):
        # out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        # out = F.log_softmax(self.fc3(out))
        # return out

# class Critic(nn.Module):

    # def __init__(self,input_size,output_size,256):
        # super(Critic, self).__init__()
        # self.fc1 = nn.Linear(input_size,hidden_size)
        # self.fc2 = nn.Linear(hidden_size,hidden_size)
        # self.fc3 = nn.Linear(hidden_size,output_size)

    # def forward(self,x):
        # out = F.relu(self.fc1(x))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)
        # return out





