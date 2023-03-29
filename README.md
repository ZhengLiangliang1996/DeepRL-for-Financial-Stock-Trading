# DeepRL-for-Financial-Stock-Trading

A  Financial stock price prediction based on Deep Reinforcement Learning in Pytorch

## Getting Started

This is a pytorch implementation tool box for Financial Stock Trading, support training with CPU/GPU. The model includes


### Prerequisites

```
git clone git@github.com:ZhengLiangliang1996/DeepRL-for-Financial-Stock-Trading.git
```

### Installing

A step by step series of examples that tell you how to run it

Prerequisites:

Python >=3.8,<3.10 (not yet 3.10)
Poetry 1.2.1+

`poetry install`

And simply run

```
chmod +x run.sh
./run.sh
```
or run the main function using python

In the result draw folder you could draw the result pickle files by 
```
python picklefile_draw_helper.py
```

## Implementation Details and NN Architecture
0. MACD
1. DQN
2. Dueling DQN
3. DDPG(Linear for actor and critic)
4. DDPG(LSTM for actor and critic)



