#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""

from __future__ import print_function
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

def draw_reward():
    all_pickles_1 = sorted(glob("wdqn_reward_list"))
    all_pickles_2 = sorted(glob("wddqn_reward_list"))
    all_pickles_3 = sorted(glob("wddpg_reward_list"))
    all_pickles_4 = sorted(glob("wddpg_lstm_reward_list"))
    # all_pickles_5 = sorted(glob("768_unit_2_layer_bilstm_forget.pickle"))
    # all_pickles_6 = sorted(glob("768_unit_2_layer_lstm_forget.pickle"))
    # # all_pickles_7 = sorted(glob("768_unit_2_layer_lstm_forget.pickle"))
    all_pickles = np.concatenate((all_pickles_1, all_pickles_2, all_pickles_3, all_pickles_4))
    #                              all_pickles_5, all_pickles_6))
    # all_pickles = all_pickles_1

    #all_pickles = np.concatenate((all_pickles_1, all_pickles_2))

    # Extract the name of each model
    model_names = [item[:-5] for item in all_pickles]

    print(model_names)
    # Extract the loss history for each model
    reward_list = [pickle.load(open(i, "rb")) for i in all_pickles]
    # Save the number of epochs used to train each model

    # Plot the training loss vs. epoch for each model
    fig = plt.figure(figsize=(4,3))
    ax1 = fig.add_subplot(111)
    colorlist = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b', '#e377c2']
    for i in range(len(all_pickles)):
        #print(reward_list)
        ax1.plot(reward_list[i], color=colorlist[i],label=model_names[i])
        # Clean up the plot
        leg1 = ax1.legend()
        leg1.get_frame().set_edgecolor('black')


    plt.xlabel('Episode')
    plt.tight_layout()
    plt.ylabel('reward')
    plt.show()

def draw_profit():
    all_pickles_1 = sorted(glob("stock__stock"))
    all_pickles_2 = sorted(glob('macd_profit'))
    all_pickles_3 = sorted(glob("wdqn_profit"))
    all_pickles_4 = sorted(glob("wddqn_profit"))
    all_pickles_5 = sorted(glob("wddpg_profit"))
    all_pickles_6 = sorted(glob("wddpg1_profit"))
    #all_pickles_6 = sorted(glob("ppo_profit"))
    # # all_pickles_7 = sorted(glob("768_unit_2_layer_lstm_forget.pickle"))
    all_pickles = np.concatenate((all_pickles_1, all_pickles_2, all_pickles_3, all_pickles_4,
                                  all_pickles_5, all_pickles_6))
    # all_pickles = all_pickles_1

    #all_pickles = np.concatenate((all_pickles_1, all_pickles_2))

    # Extract the name of each model
    model_names = [item[:-7] for item in all_pickles]

    print(model_names)
    # Extract the loss history for each model
    reward_list = [pickle.load(open(i, "rb")) for i in all_pickles]
    # Save the number of epochs used to train each model

    # Plot the training loss vs. epoch for each model
    fig = plt.figure(figsize=(4,3))
    ax1 = fig.add_subplot(111)
    colorlist = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b', '#e377c2']
    for i in range(len(all_pickles)):
        # print(reward_list)
        ax1.plot(reward_list[i], color=colorlist[i],label=model_names[i])
        # Clean up the plot
        leg1 = ax1.legend()
        leg1.get_frame().set_edgecolor('black')


    plt.xlabel('Time(day)')
    plt.tight_layout()
    plt.ylabel('profit percentage')
    plt.show()


def main():
    draw_reward()
    draw_profit()

if __name__ == "__main__":
    main()

