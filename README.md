# Continuous Control

<img src="./img/tennis.png" width="500" />

This notebook presents the work of training two Reinforcement Learning (RL) agents to control rackets to bounce a ball over a net, in the Unity ML-Agents environment. The model that based on the Deep Deterministic Policy Gradient(DDPG) algorithm.


### Introduction
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

### Solving the Environment
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Python environment
Python 3.6 is used to conduct the work. However, Python 3.5 should also work.

To prepare the Python environment, first create a virtual environment using your favoured package mangement tool, then `cd` to the repo's root directory and run the following command:

```
pip -q install ./python
```

### Download the Unity environment
For this project, you will not need to install Unity - this is because we (Udacity) have already built the environment for you, and you can download it from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, unzip (or decompress) the file to the repo's root directory.

### Train the agent
In `Report.ipynb`, it details the steps on how to train the agent from scratch. You don't need to have a GPU to train it.

### References
- Udacity Deep Reinforcement Learning course and [repo](https://github.com/udacity/deep-reinforcement-learning)
- [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971)