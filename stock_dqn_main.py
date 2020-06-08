#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""
import random
import gym
import sys
import random
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from env.StockEnv import StockEnv
from model.network import DQN, DQN_1, DQN_2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5, help='number of epoches to run')
parser.add_argument('--reward_name', type=str, default='dqn_reward', help='reward list name will be pickled')
parser.add_argument('--backtest_profit', type=str, default='profit', help='backtest profit name will be pickled')
parser.add_argument('--backtest_action', type=str, default='backtest_action', help='backtest profit name will be pickled')
parser.add_argument('--ddqn', type=bool, default=False, help='Whether to use dueling DQN')



args = parser.parse_args()


# set it to 1 or 0 when loading the checkpoint
EPISODES = args.epochs
# set it to as small as possible when loading the checkpoint eg: 0.001
EPSILON = 0.9
GAMMA = 0.9
LEARNING_RATE = 0.02
REPLACE_TARGET_STEP = 5
START_LEARNING_STEP = 200
TIME_FRAME = 90
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def argmax(l):
    """ Return the index of the maximum element of a list
    """
    return max(enumerate(l), key=lambda x:x[1])[0]

class BufferMemory(object):
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.memory = []
        self.batch_size = batch_size

    # push experience into the memory list
    def push_SSRA(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # sample experience from the memory list
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)

        # transfer the numpy array to tensor
        next_states = torch.from_numpy(np.vstack([e[0] for e in experiences if e])).float().to(device)
        states      = torch.from_numpy(np.vstack([e[1] for e in experiences if e])).float().to(device)
        actions     = torch.from_numpy(np.vstack([e[2] for e in experiences if e])).long().to(device)
        rewards     = torch.from_numpy(np.vstack([e[3] for e in experiences if e])).float().to(device)
        terminates  = torch.from_numpy(np.vstack([e[4] for e in experiences if e]).astype(np.uint8)).float().to(device)
        return (next_states, states, actions, rewards, terminates)

    def size(self):
        return len(self.memory)



class Agent(object):
    def __init__(self, action_space, state_space, episilon, learning_rate, discount):
        self.episilon       = episilon
        self.learning_rate  = learning_rate
        self.replay_buffer  = BufferMemory(1000, 200)
        self.action_space   = action_space
        self.state_space    = state_space
        self.update_step    = REPLACE_TARGET_STEP
        self.t_step         = 0
        if not args.ddqn:
            self.network        = DQN_1(self.state_space, self.action_space).to(device)
            self.target_network = DQN_1(self.state_space, self.action_space).to(device)
        else:
            print('aaa')
            self.network        = DQN_2(self.state_space, self.action_space).to(device)
            self.target_network = DQN_2(self.state_space, self.action_space).to(device)
        self.optimizer      = optim.Adam(self.network.parameters(), lr = self.learning_rate)
        self.discount       = discount
        self.tau            = 1e-3

    def train_network(self, experiences):
        (next_states, states, actions, rewards, terminates) = experiences

        # Estimation
        q_estimation = self.network(states).gather(1, actions)
        # Target
        q_target = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)

        # rewards + discount * max(max_q_value
        Y_i = rewards + self.discount * q_target * (1 - terminates)

        # Compute the loss
        loss = F.mse_loss(q_estimation, Y_i)
        # print(loss)
        self.optimizer.zero_grad()
        # backward gradient
        loss.backward()
        self.optimizer.step()

        # update the newtork
        self.update_network(self.target_network, self.network)

    # Update model
    def update_network(self, target_network, network):
        """
        update the model parameters
        copy from the estimation_theta * tau + (1 - tau) * target_theta
        """
        for i, j in zip(self.target_network.parameters(), self.network.parameters()):
            i.data.copy_(self.tau * j.data + (1.0 - self.tau) * i.data)

    # Exploration Strategy
    def epsilon_greedy(self, state):
        # generate a random probability and compare it with eps
        rand_p = np.random.rand()

        # Explore
        if rand_p < self.episilon:
            return np.random.randint(0, self.action_space)
        # Exploitation
        else:
            with torch.no_grad():
               # t.max(1) will return largest column value of each row.
               # second column on max result is index of where max element was
               # found
                #print(state)
                s0 = torch.tensor([state], device=device, dtype=torch.float)
                # expand dimension
                #s0 = s0.unsqueeze(0)
                a1 = self.network(s0)
                print(a1)
                a = self.network(s0).max(1)[1].view(1, 1)
                return a.item()


    def agent_step(self, next_state, state, action, reward, terminate):
        transition = (next_state, state, action, reward, terminate)
        self.replay_buffer.push_SSRA(transition)

        self.t_step += 1

        if self.t_step > START_LEARNING_STEP and (self.t_step % self.update_step == 0):
            if self.replay_buffer.size() > self.replay_buffer.batch_size:

                experiences = self.replay_buffer.sample()

                self.train_network(experiences)

def run_single_episode(agent, env, state, log=False, strategy=None):
    # Choose an action at random
    action = agent.epsilon_greedy(state)

    # Perform the action
    next_state, r, terminate = env.step(action, is_logging = log)

    # Update the value and do the experience replay
    agent.agent_step(next_state, state, action, r, terminate)

    #Update satistics
    # cumulative_reward += r
    state = next_state

    return state, terminate, r


def backtest(env, agent, log=False):
    state = env.reset()
    terminate = False

    while not terminate:
        next_states, terminate, r = run_single_episode(agent, env, state, log)

    # agent.episilon = max(eps_lowest, eps_decay*agent.episilon)
    return env


def main():
    #read data from csv file
    stock_data = '000065.SZ_NormalData.csv'
    df         = pd.read_csv(stock_data)
    df         = df.sort_values('trade_date', ascending = True)
    # clean data, drop those without stock moving averages
    df         = df.iloc[22:].reset_index(drop=True)

    # Create the stock envronment
    env = StockEnv(df.iloc[0:1500],3, TIME_FRAME, TIME_FRAME)
    print(env.action_space)
    print(env.state_space)
    action_space = env.action_space
    state_space  = env.state_space

    # Act randomly in the environment
    average_cumulative_reward = 0.0

    eps_lowest = 0.01
    eps_decay = 0.995
    # Create Agent
    agent = Agent(action_space, state_space, EPSILON, LEARNING_RATE, GAMMA)

    # reward list
    rewards_list = []

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0
        # Loop over time-steps
        while not terminate:
            state, terminate, reward = run_single_episode(agent, env, state, log=True)
            # Episilon Decay
            cumulative_reward += reward
        agent.episilon = max(eps_lowest, eps_decay*agent.episilon)
        # Per-episode statistics
        average_cumulative_reward *= 0.95
        average_cumulative_reward += 0.05 * cumulative_reward

        rewards_list.append(average_cumulative_reward)

        #print(i, cumulative_reward, average_cumulative_reward)
        #print("="*80)
        # if average_cumulative_reward >= 200:
            # torch.save(agent.network.state_dict(), 'dqn.pth')

    # pickle the reward in a list
    import pickle
    with open(args.reward_name, 'wb') as f:
        pickle.dump(rewards_list, f)

    # BackTest
    agent.episilon = 0
    env_backtest = StockEnv(df.iloc[1500:].reset_index(drop=True), 3, TIME_FRAME, TIME_FRAME)
    env_backtest = backtest(env_backtest, agent, True)
    a = args.backtest_action
    b = args.backtest_profit
    env_backtest.render(a,b)


if __name__ == '__main__':
    main()
