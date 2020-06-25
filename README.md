# DeepRL-for-Financial-Stock-Trading

A  Financial stock price prediction based on Deep Reinforcement Learning in Pytorch

## Getting Started

This is a pytorch implementation tool box for Financial Stock Trading, support training with CPU/GPU. The model includes


### Prerequisites

First you need to install Pytorch 1.4.0

And then

```
git clone git@github.com:ZhengLiangliang1996/DeepRL-for-Financial-Stock-Trading.git
```

### Installing

A step by step series of examples that tell you how to run it

Simply run

```
chmod +x run.sh
./run.sh
```
or run the main function using python

In the result draw folder you could draw the result pickle files by 
```
python picklefile_draw_helper.py
```
I used vsc_stimu3_brnn.pbs file for running in VUB VSC, if anyone wants to run it in a super computer center, feel free to use that, the code is almost the same. simply run by

```
qsub vsc_stimu3_brnn.pbs
```

## Implementation Details and NN Architecture
0. MACD
1. DQN
2. Dueling DQN
3. DDPG(Linear for actor and critic)
4. DDPG(LSTM for actor and critic)



