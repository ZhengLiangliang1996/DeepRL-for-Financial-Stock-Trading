#!/bin/sh
poetry run python stock_dqn_main.py --epochs=500 --reward_name='dqn_reward_list' --backtest_profit='dqn_profit' --backtest_action='dqn_action' 
poetry run python stock_dqn_main.py --epochs=500 --reward_name='ddqn_reward_list' --backtest_profit='ddqn_profit' --backtest_action='ddqn_action' --ddqn=True
portry run python stock_ddpg_main.py --epochs=500 --reward_name='ddpg_reward_list' --backtest_profit='ddpg_profit' --backtest_action='ddpg_action' 
portry run python stock_ddpg_main.py --epochs=500 --reward_name='ddpg_lstm_reward_list' --backtest_profit='ddpg_lstm_profit' --backtest_action='ddpg_lstm_action' --ddpg_lstm=True

