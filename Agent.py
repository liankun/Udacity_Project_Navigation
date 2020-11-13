import numpy as np
import random
from collections import namedtuple,deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from Model import QNetWork

class ReplayPool:
    """
    Fixed-size buffer to store experience tuples
    """
    def __init__(self,buffer_size,batch_size,device='cpu'):
        """
        action_size: size of action space
        buffer_size: maximum size of buffer
        batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",field_names=["state","action","reward","next_state","done"])
        self.device = device
        
    def add(self,state,action,reward,next_state,done):
        """
        add a new experience to memory
        """
        e = self.experience(state,action,reward,next_state,done)
        self.memory.append(e)
    
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        """
        experiences = random.sample(self.memory,k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return (states,actions,rewards,next_states,dones)
    
    
    def __len__(self):
        """
        return the current size of internal memory
        """
        return len(self.memory)
    

class Agent():
    """
    interact with the environment 
    """
    def __init__(self,state_size,
                 action_size,
                 replay_pool_size=int(1e5),
                 batch_size=64,
                 gamma=0.99,
                 update_step=4,
                 lr=5e-4,
                 tau=1e-3,
                 device='cpu'):
        """
        state_size: the size of state space
        action_size: the size of actoin space
        replay_pool_size: the size of replay size need to store
        batch_size: the size of minibatch used in learning
        gamma:discount rate
        update_step: how often to update target network
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device
        
        #QNetWork
        self.qnetwork_local = QNetWork(state_size,action_size).to(self.device)
        self.qnetwork_target = QNetWork(state_size,action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=lr)
        
        #create replay pool
        self.memory = ReplayPool(replay_pool_size,batch_size,device=self.device)
        
        self.t_step = 0
        self.update_step = update_step
        
    def step(self,state,action,reward,next_state,done):
        #put the experience into the pool
        self.memory.add(state,action,reward,next_state,done)
        
        #learn every update_step
        self.t_step = (self.t_step + 1)%self.update_step
        if self.t_step ==0:
            # if enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def act(self,state,eps=0.):
        """
        return actions for given state as per current policy
        state : current state
        eps: epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            actio_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        #epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(actio_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    def learn(self,experiences):
        """
        update the qnetwork local
        experiences: tuple of (s,a,r,s',done) tuples
        gamma (float): discount factor
        """
        states,actions,rewards,next_states,dones = experiences
        
        self.optimizer.zero_grad()
        q_local = self.qnetwork_local.forward(states).gather(1,actions)
        q_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_target = rewards+self.gamma*q_next*(1-dones)
        loss = F.mse_loss(q_target,q_local)
        loss.backward()
        self.optimizer.step()
        
        #soft update of target network
        self.soft_update()
        
    def soft_update(self):
        """
        weight_target = tau*weight_local+(1-tau)*weight_target
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(),self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data+(1-self.tau)*target_param.data)      

    
