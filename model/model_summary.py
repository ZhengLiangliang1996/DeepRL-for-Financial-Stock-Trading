#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from network import DQN_1, DQN_2

def main():
    #device = torch.device("cpu") # PyTorch v0.4.0
    model = DQN_1(90, 3)
    summary(model, (1, 1, 90))

if __name__ == "__main__":
    main()

