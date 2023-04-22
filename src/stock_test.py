#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""

from __future__ import print_function
import sys
import argparse

import random
import gym
import sys
import random
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from env.StockEnv import StockEnv
from model.network_A2C import A2C

from collections import namedtuple
from torch.distributions import Categorical

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()
        # self.bn1     = nn.BatchNorm(90)
        self.affine1 = nn.Linear(6, 128)

        # actor's layer
        self.action_head = nn.Linear(128, 3)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        # x = self.bn1(x)
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim = 0)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

class Skylark_A2C():
    def __init__(self):
        self.model = Policy()
        stock_data = '000065.SZ_NormalData.csv'
        df         = pd.read_csv(stock_data)
        df         = df.sort_values('trade_date', ascending = True)
        # clean data, drop those without stock moving averages
        df         = df.iloc[22:].reset_index(drop=True)

        self.env = StockEnv(df.iloc[0:1500],3, 6, 6)

        self.optimizer = optim.Adam(self.model.parameters(), lr = 3e-2)
        self.eps = np.finfo(np.float32).eps.item()
        self.SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
        self.render = False
        self.log_interval = 1
        self.gamma = 0.99

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        print(probs)
        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()
        print(probs)

        # save to action buffer
        self.model.saved_actions.append(self.SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.model.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        # gradient clipping
        #nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)

        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]

    def train(self, num_episodes):
        running_reward = 10

        # run inifinitely many episodes
        for i in range(1, num_episodes):

            # reset environment and episode reward
            state = self.env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            for t in range(1, 10000):
                # select action from policy
                action = self.select_action(state)

                # take the action
                state, done, reward = self.env.step(action, is_logging = True)

                if self.render:
                    self.env.render()

                self.model.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            self.finish_episode()

            # log results
            if i % self.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i, ep_reward, running_reward))

            # check if we have "solved" the cart pole problem
            #if running_reward > self.env.spec.reward_threshold:
            #    print("Solved! Running reward is now {} and "
            #        "the last episode runs to {} time steps!".format(running_reward, t))
            #    break


if __name__ == "__main__":
        ac_agent = Skylark_A2C()
        ac_agent.train(100)
