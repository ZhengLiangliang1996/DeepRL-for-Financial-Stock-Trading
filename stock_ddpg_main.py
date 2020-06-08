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
from model.network_DDPG import Actor, Critic, Actor1, Critic1, Actor2, Critic2


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1, help='number of epoches to run')
parser.add_argument('--reward_name', type=str, default='dqn_reward', help='reward list name will be pickled')
parser.add_argument('--backtest_profit', type=str, default='profit', help='backtest profit name will be pickled')
parser.add_argument('--backtest_action', type=str, default='backtest_action', help='backtest profit name will be pickled')
parser.add_argument('--ddpg_lstm', type=bool, default=False, help='Whether to use dueling DQN')



args = parser.parse_args()


# set it to 1 or 0 when loading the checkpoint
EPISODES = args.epochs
# set it to as small as possible when loading the checkpoint eg: 0.001
EPSILON = 0.9
GAMMA = 0.99
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
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
        actions     = torch.from_numpy(np.vstack([e[2] for e in experiences if e])).float().to(device)
        rewards     = torch.from_numpy(np.vstack([e[3] for e in experiences if e])).float().to(device)
        terminates  = torch.from_numpy(np.vstack([e[4] for e in experiences if e]).astype(np.uint8)).float().to(device)

        return (next_states, states, actions, rewards, terminates)

    def size(self):
        return len(self.memory)



class Agent(object):
    def __init__(self, action_space, state_space, episilon, actor_learning_rate,
                 critic_learning_rate, discount):
        self.episilon       = episilon
        self.actor_learning_rate  = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.replay_buffer  = BufferMemory(500, 20)
        self.action_space   = action_space
        self.state_space    = state_space
        if not args.ddpg_lstm:
            self.actor          = Actor1(self.state_space, self.action_space, 128)
            self.actor_target   = Actor1(self.state_space, self.action_space, 128)
            self.critic         = Critic1(self.state_space+self.action_space, self.action_space, 128)
            self.critic_target  = Critic1(self.state_space+self.action_space, self.action_space, 128)
        else:
            self.actor          = Actor2(self.state_space, self.action_space, 128)
            self.actor_target   = Actor2(self.state_space, self.action_space, 128)
            self.critic         = Critic2(self.state_space+self.action_space, self.action_space, 128)
            self.critic_target  = Critic2(self.state_space+self.action_space, self.action_space, 128)

        self.update_step    = REPLACE_TARGET_STEP
        self.t_step         = 0
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_learning_rate)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_learning_rate)
        self.discount       = discount
        self.tau            = 0.02

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def train_network(self, experiences):
        (next_states, states, actions, rewards, terminates) = experiences
        # print(states.size())
        def critic_learn():
            a1 = self.actor_target(states).detach()
            y_true = rewards + self.discount * self.critic_target(next_states, a1).detach()
            y_pred = self.critic(states, actions)

            loss = F.mse_loss(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(states, self.actor(states)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
        #print(states.size())
        critic_learn()
        actor_learn()
        self.update_network(self.critic_target, self.critic)
        self.update_network(self.actor_target, self.actor)

        #update the newtork
        # self.update_network(self.target_network, self.network)

    # Update model
    def update_network(self, net_target, net):
        for target_param, param  in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    # Exploration Strategy
    def epsilon_greedy(self, state):
        # generate a random probability and compare it with eps
        rand_p = np.random.rand()

        # Explore
        if rand_p < self.episilon:
            s0 = torch.tensor([state], device=device, dtype=torch.float)
            a0 = self.actor(s0)
            return np.random.randint(0, self.action_space), a0
        # Exploitation
        else:
            # state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            # action = self.actor(state).squeeze(0).detach().numpy()
            # return action

            with torch.no_grad():
                s0 = torch.tensor([state], device=device, dtype=torch.float)
                a0 = self.actor(s0)
                a = self.actor(s0).max(1)[1].view(1, 1)
                return a.item(), a0


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
    action, a0 = agent.epsilon_greedy(state)

    # Perform the action
    next_state, r, terminate = env.step(action, is_logging = log)

    # Update the value and do the experience replay
    agent.agent_step(next_state, state, a0.data.numpy(), r, terminate)

    #Update satistics
    # cumulative_reward += r
    state = next_state

    return state, terminate, r


def backtest(env, agent, log=False):
    state = env.reset()
    terminate = False

    while not terminate:
        next_states, terminate, _ = run_single_episode(agent, env, state, log)


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
    agent = Agent(action_space, state_space, EPSILON, ACTOR_LEARNING_RATE,
                  CRITIC_LEARNING_RATE, GAMMA)

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
    import pickle
    with open(args.reward_name, 'wb') as f:
        pickle.dump(rewards_list, f)


    # BackTest
    env_backtest = StockEnv(df.iloc[1500:].reset_index(drop=True), 3, TIME_FRAME, TIME_FRAME)
    env_backtest = backtest(env_backtest, agent, True)
    a = args.backtest_action
    b = args.backtest_profit
    env_backtest.render(a,b)



if __name__ == '__main__':
    main()
