#! /usr/bin/env python
"""
Author: LiangLiang ZHENG
Date:
File Description
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

INITIAL_ACCOUNT_MONEY = 10000

class StockEnv(object):
    def __init__(self, df, action_space, state_space, window_size):
        # dataframe of the stock data
        self.df                         = df
        # buy sell and hold
        self.action_space               = action_space
        self.state_space                = state_space
        self.window_size                = window_size
        self.init_money                 = INITIAL_ACCOUNT_MONEY
        # data when today's markey is closed
        self.market_close               = df['close'].values

        self.window_size                = window_size
        self.half_window                = window_size // 2
        # tax: according to Chinese Stock Markey Standard
        self.buy_service_rate           = 0.0003
        self.sell_service_rate          = 0.0003
        self.buy_min_trans              = 5
        self.sell_min_trans             = 5
        self.stamp_tax_rate             = 0.001

    # TODO: explain frame d in the report
    def next_observation(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1

        frame = []
        if d < 0:
            for i in range(-d):
                # when days is not enough, padding with 0
                frame.append(self.market_close[0])
            for i in range(t+1):
                frame.append(self.market_close[i])
        else:
            frame = self.market_close[d:t+1]

        res = []
        for i in range(window_size - 1):
            res.append(frame[i + 1] - frame[i] / (frame[i] + 0.0001))

        return np.array(res)

    def reset(self):
        # money in the account
        self.money_hold                 = self.init_money
        # number of stock held
        self.stock_hold                 = 0
        # number bought
        self.buy_num                    = 0
        # total stock value
        self.total_stock_value          = 0
        # total market value (+money you have)
        self.total_market_value         = 0
        # yesterday toatl market value
        self.yesterday_market_value     = self.init_money
        # total profit
        self.total_profit               = 0
        # time stamp: used for getting specific observation
        self.t                          = self.state_space // 2
        # reward
        self.reward                     = 0

        # time of sold
        self.state_sold                = []
        # time of bought
        self.state_bought               = []

        # profit made
        self.profit_percentage_account  = []
        self.profit_percentage_stock    = []
        return self.next_observation(self.t)

    def stock_buy(self):
        """Stock buying
        """
        self.buy_num                    =  self.money_hold // self.market_close[self.t] // 100
        self.buy_num                    = self.buy_num * 100

        # calculating how much tax and service fee
        before_amount                   = self.market_close[self.t] * self.buy_num
        service_amount                  = before_amount * self.buy_service_rate
        # should be higher than the minimum transaction fee
        # otherwise go with the loweest
        service_amount                  = max(service_amount, self.buy_min_trans)

        total_amount                    = before_amount + service_amount
        if total_amount > self.money_hold:
            self.buy_num -= 100

        # recount the service amount
        before_amount  = self.market_close[self.t] * self.buy_num
        service_amount = before_amount * self.buy_service_rate
        service_amount = max(service_amount, self.buy_min_trans)

        # update
        self.stock_hold                 += self.buy_num
        self.total_stock_value          += self.market_close[self.t] * self.buy_num
        total_amount                     = before_amount + service_amount
        self.money_hold                 -= total_amount
        self.state_bought.append(self.t)

    # TODO: sell it early
            #止盈 和 止损
    def stock_sold(self, num):
        """stock sold
        """
        before_amount                   = num * self.market_close[self.t]
        service_amount                  = before_amount * self.sell_service_rate
        if service_amount < self.sell_min_trans:
            service_amount = self.sell_min_trans
        stamp_tax = self.stamp_tax_rate * before_amount

        self.money_hold = self.money_hold + before_amount - service_amount - stamp_tax
        # what if the agent only sold part of the stock?
        self.stock_hold = 0
        self.total_stock_value = 0
        self.state_sold.append(self.t)

    # def strategy1(self):
        # 止损止盈



    def strategy(self):
        # pure price comparison
        if self.df['close'][self.t] > self.df['ma21'][self.t]:
            return True
        else:
            return False
    def logger(self, action):
        if action == 1:
            print('bought price:{} stock num:{} stock hold now:{} cash {:.4} \
                    day:{:d}'.format(self.market_close[self.t], self.buy_num,\
                    self.stock_hold, self.money_hold,self.t))
        elif action == 2:
            print('Sold price:{} stok hold now:{} cash: {:.4} day:{:d}'.format( \
                    self.market_close[self.t], self.stock_hold, self.money_hold, self.t))
        else:
            print('hold and do nothing')


    def step(self, action, is_logging):
        """ stock step
            action: the action took in the env
            return: next_state, reward, termination
        """
        # buy stock, at least 100, and the length of the day is enough
        if action == 1 and self.money_hold >= (self.market_close[self.t]*100 + max(self.buy_min_trans, self.market_close[self.t] * 100 * self.buy_service_rate)):

            BuyFlag = True
            # strategy
            if not self.strategy():
                BuyFlag = False

            if BuyFlag:
                self.stock_buy()
                if is_logging:
                    self.logger(action)

        # sell the stock
        elif action == 2 and self.stock_hold > 0:
            self.stock_sold(self.stock_hold)
            if is_logging:
                self.logger(action)

        else:
            # hold the stock
            self.logger(0)

        # calculate the profit and the reward needed to be returned
        self.total_stock_value       = self.market_close[self.t] * self.stock_hold

        self.total_market_value      = self.total_stock_value + self.money_hold
        self.total_profit            = self.total_market_value - self.init_money

        # calculate the reward
        reward = (self.market_close[self.t + 1] - self.market_close[self.t]) / self.market_close[self.t]

        if np.abs(reward) <= 0.015:
            self.reward = reward * 0.2
        elif np.abs(reward) <= 0.03:
            self.reward = reward * 0.7
        elif np.abs(reward) >= 0.05:
            if reward < 0 :
                self.reward = (reward + 0.05) * 0.1 - 0.05
            else:
                self.reward = (reward - 0.05) * 0.1 + 0.05

        if self.stock_hold > 0 or action == 2:
            self.reward = reward
            if action == 2:
                self.reward = -self.reward
        else:
            self.reward = -self.reward * 0.1

        self.yesterday_market_value = self.total_market_value

        self.profit_percentage_account.append((self.total_market_value - self.init_money) / self.init_money)
        self.profit_percentage_stock.append((self.market_close[self.t] - self.market_close[0]) / self.market_close[0])

        terminate = False
        self.t += 1
        if self.t == len(self.market_close) - 2:
            terminate = True

        next_state = self.next_observation(self.t)
        reward = self.reward

        return next_state, reward, terminate


    def render(self, name1, name2):
        """Rendering the graph after training
        """

        fig = plt.figure(figsize=(20, 5))
        print(np.shape(self.market_close))
        print(np.shape(self.state_sold))
        plt.plot(self.market_close, color='b', linewidth=2)
        plt.plot(self.market_close, 'v', markersize=8,color='r', label='sold signal', markevery = self.state_sold)
        plt.plot(self.market_close, '^', markersize=8,color='g', label='bought signal', markevery =self.state_bought)
        #plt.title('total profit%f', self.total_profit)
        plt.xlabel("Time(day)")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

        plt.savefig(name1+'.pdf')
        plt.close()

        import pickle
        with open(name2, 'wb') as f:
            pickle.dump(self.profit_percentage_account, f)

        with open('backtest_stock', 'wb') as f:
            pickle.dump(self.profit_percentage_stock, f)


#         fig = plt.figure(figsize=(20, 5))
        # plt.plot(self.profit_percentage_account, label='Account')
        # plt.plot(self.profit_percentage_stock, label='Stock')
        # plt.legend()
        # plt.show()
#         #plt.savefig(name2)
        #plt.close()





