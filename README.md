[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

For this project, an agent is trained to navigate (and collect bananas!) in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas. It is considered solved when the rewards is bigger than 13. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

    
### Instructions
The Project is done in windows 64bit environment. 
- Navigation.ipynb : The main code to train and evalue the agent
- Agent.py : the code of ReplayPool and Agent
- Model.py : the code of DQN
- checkpoint.pth : the trained DQN weight
- Report.ipynb : report of the project
- trained_agent.mp4 : shows a performance of a trained agent
- Banana_Windows_x86_64 : The unity environment for the project 

Necessary Packets:
- python 3.6.9
- numpy 1.17.0
- pytorch 1.0.1 py3.6_cuda100_cudnn7_1
- unityagents 0.4.0


