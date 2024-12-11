import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Optimizer import init_weights
import gymnasium as gym
import ale_py



class ActorCritic(nn.Module):
    def __init__(self, input_channels, action_space):
        super().__init__()
        
        # Convolutional layers for A3C Network
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # LSTM for temporal dependencies
        self.lstm = nn.LSTMCell(32 * 3 * 3, 256)

        # Output layers
        num_outputs = action_space.n
        #Critic Output
        self.critic = nn.Linear(256, 1)
        #Actor Output
        self.actor = nn.Linear(256, num_outputs)

        #Initialization
        self.apply(init_weights)
        self._reset_actor_critic_weights()



    def _reset_actor_critic_weights(self):
        # Custom initialization for actor
        std_actor = 0.01
        actor_out = torch.randn(self.actor.weight.data.size())
        self.actor.weight.data = actor_out * std_actor / torch.sqrt(actor_out.pow(2).sum(1, keepdim=True))
        self.actor.bias.data.fill_(0)
        
        # Custom initialization for critic
        std_critic = 1.0
        critic_out = torch.randn(self.critic.weight.data.size())
        self.critic.weight.data = critic_out * std_critic / torch.sqrt(critic_out.pow(2).sum(1, keepdim=True))
        self.critic.bias.data.fill_(0)

        # Zero-Initialize LSTM biases 
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)



    def forward(self, inputs):
        state, (hx, cx) = inputs
        x = F.elu(self.conv1(state))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        #Flatten LSTM input
        x = x.view(-1, 32 * 3 * 3)
        
        hx, cx = self.lstm(x, (hx, cx))

        # return predcitions
        return self.critic(hx), self.actor(hx), (hx, cx)
