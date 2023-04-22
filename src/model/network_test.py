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

class A2C(nn.Module):
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


    def forward(self, states):
        x = states.view(-1, 1, self.state_space)

