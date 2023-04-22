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
from env.StockEnv_MACD import StockEnv
from model.network import DQN, DQN_1, DQN_2

# set it to 1 or 0 when loading the checkpoint
EPISODES = 5
# set it to as small as possible when loading the checkpoint eg: 0.001
EPSILON = 0.9
GAMMA = 0.9
LEARNING_RATE = 0.02
REPLACE_TARGET_STEP = 5
START_LEARNING_STEP = 200
TIME_FRAME = 90
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def backtest(env, agent, log=False):
    state = env.reset()
    terminate = False

    while not terminate:
        next_states, terminate = run_single_episode(agent, env, state, log)

    # agent.episilon = max(eps_lowest, eps_decay*agent.episilon)
    return env

# def calculateEMA(period, closeArray, emaArray=[]):
    # """计算指数移动平均"""
    # length = len(closeArray)
    # nanCounter = np.count_nonzero(np.isnan(closeArray))
    # if not emaArray:
        # emaArray.extend(np.tile([np.nan],(nanCounter + period - 1)))
        # firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])
        # emaArray.append(firstema)
        # for i in range(nanCounter+period,length):
            # ema=(2*closeArray[i]+(period-1)*emaArray[-1])/(period+1)
            # emaArray.append(ema)
    # return np.array(emaArray)

# def calculateMACD(closeArray,shortPeriod = 12 ,longPeriod = 26 ,signalPeriod =9):
    # ema12 = calculateEMA(shortPeriod ,closeArray,[])
    # ema26 = calculateEMA(longPeriod ,closeArray,[])
    # diff = ema12-ema26

    # dea= calculateEMA(signalPeriod ,diff,[])
    # macd = 2*(diff-dea)
    # return macd,diff,dea

def calculateEMA(period, closeArray, emaArray=[]):
    """计算指数移动平均"""
    length = len(closeArray)
    nanCounter = np.count_nonzero(np.isnan(closeArray))
    if not emaArray:
        emaArray.extend(np.tile([np.nan],(nanCounter + period - 1)))
        firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])
        emaArray.append(firstema)
        for i in range(nanCounter+period,length):
            ema=(2*closeArray[i]+(period-1)*emaArray[-1])/(period+1)
            emaArray.append(ema)
    return np.array(emaArray)

def calculateMACD(closeArray,shortPeriod = 12,longPeriod = 26 ,signalPeriod =9):
    ema12 = calculateEMA(shortPeriod ,closeArray,[])
    ema26 = calculateEMA(longPeriod ,closeArray,[])
    diff = ema12-ema26

    dea= calculateEMA(signalPeriod ,diff,[])
    macd = 2*(diff-dea)
    return macd,diff,dea


def get_signal(array):
    hist, diff, dea = calculateMACD(array)
    sig1 = [0] * len(array)
    sig2 = [0] * len(array)
    for i in range(len(hist)):
        if not np.isnan(hist[i]):
            sig1[i] = 1 if hist[i] > 0 else 2
        if np.isnan(diff[i]):
            diff[i] = 0
        if np.isnan(dea[i]):
            dea[i] = 0
    for i in range(len(hist)):
        if not np.isnan(hist[i]):
            sig2[i] = hist[i]
    return sig1, sig2, diff, dea

def main():
    #read data from csv file
    stock_data = '000065.SZ_NormalData.csv'
    df         = pd.read_csv(stock_data)
    df         = df.sort_values('trade_date', ascending = True)
    # clean data, drop those without stock moving averages
    df         = df.iloc[22:].reset_index(drop=True)

    # Create the stock envronment
    env = StockEnv(df.iloc[1500:].reset_index(drop=True),3, TIME_FRAME, TIME_FRAME)

    # Act randomly in the environment
    average_cumulative_reward = 0.0

    eps_lowest = 0.01
    eps_decay = 0.995
    # Create Agent

    # reward list
    rewards_list = []

    # Loop over episodes
    #for i in range(EPISODES):
    state = env.reset()
    terminate = False
    d = df.iloc[1500:]['close'].values
    print(np.shape(d))
    actions, hist, dif, dea = get_signal(d)
    for i in range(len(actions)-90):
        _, r, t = env.step(actions[i], is_logging = True)

    env.render("1.png", "macd_profit")
    fig = plt.figure()
    ax = plt.subplot(111)
    x = range(1, len(hist) + 1)
    print(len(dif))
    ax.bar(x, hist, width=1, color='r', label='MACD')
    plt.plot(dif, color='b',linewidth=1, label='diff')
    plt.plot(dea, color='g', linewidth=1, label='DEA')
    plt.xlabel('Time(day)')
    plt.ylabel('Moving Average')
    plt.legend()
    plt.show()
    # cumulative_reward = 0.0
        # Loop over time-steps
        # Episilon Decay
        # Per-episode statistics
        #average_cumulative_reward *= 0.95
        #average_cumulative_reward += 0.05 * cumulative_reward

        #rewards_list.append(average_cumulative_reward)

        #print(i, cumulative_reward, average_cumulative_reward)
        #print("="*80)
        # if average_cumulative_reward >= 200:
            # torch.save(agent.network.state_dict(), 'dqn.pth')


    # BackTest
    # env_backtest = StockEnv(df.iloc[1500:].reset_index(drop=True), 3, TIME_FRAME, TIME_FRAME)
    # d = df.iloc[0:1500]['close'].values
    # print(np.shape(d))
    # actions = get_signal(d)
    # print(actions)

    # env_backtest = backtest(env_backtest, agent, True)
    # env_backtest.render('training.png', 'profit.png')


if __name__ == '__main__':
    main()
