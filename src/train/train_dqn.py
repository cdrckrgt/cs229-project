'''
Cedrick Argueta

some of code taken from:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import random
import math
import numpy as np
from itertools import count
from collections import namedtuple
import psutil
import yaml
import os
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from util import soft_update, hard_update, log_weights

from PyFEBOLEnv import PyFEBOLEnv

'''
configuration
-------------

code in this section does multiple things for setting up the experiment.

things to note:
    creating the yaml file here is essential to setting up reproducible experimental
        conditions. training, testing, and baselines depend on this yaml file.

    the environment (PyFEBOL) mostly conforms to the OpenAI gym environment
        specifications.

    when changing between sensor modalities (or any configurable stuff in general),
        make sure to comment/uncomment the necessary parameters. you can find which
        ones are necessary in PyFEBOLEnv.py
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on: ", device)

run = 'run_02'

writer = SummaryWriter('../../logs/{}'.format(run))

config = {
    'run' : run,

    # search domain config
    'domain_length' : 200,
    'target_max_step' : 1.7,
    'nb_directions' : 12, # move in 30 degree radially spaced
    'nb_max_episode_steps' : 100,
    'nb_act_repeat' : 1,
    'action_dimensionality' : 'discrete',

    # cost model config
    'cost_model' : 'weighted_threshold',
    # collision
    'lambda_1' : 1,
    # entropy/belief
    'lambda_2' : 1,
    # tracking error
    'lambda_3' : 0,
    'distance_threshold' : 15,
    'entropy_threshold' : 0,
    'tracking_threshold' : 0,

    # sensor config
    # if doing bearing_only, then only sensor_sd must be set
    # if doing fov sensor, sensor alpha, cone width, blind distance, and
    # headings must be set
    'sensor_type' : 'bearing_only',
    'sensor_sd' : 5.0,

    # filter config
    'filter_type' : 'particle',
    'filter_length' : 50,
    'nb_particles' : 2000,

    # training hyperparams
    'nb_episodes' : int(1e5), # nb_episodes * nb_max_episode_steps / nb_act_repeat = total steps
    'eps_start' : 0.9,
    'eps_end' : 0.1,
    # this is nb_episodes * max_ep_length /nb_act_repeat / 6
    'eps_decay' : int(1e5) / 6, # this number times six is where we reach about 0.1 in steps
    'gamma' : 0.995, # 1 / (1 - gamma) is timescale?
    'lr' : 5e-4,
    'batch_size' : 128,
    'target_update' : 5e-4, 
    'memory_size' : int(1e5), # warning: gets bigger with phi_length bigger
    'nb_warmup_steps' : 1000,
    'use_double_dqn' : True,
    'phi_length' : 1, # how many ticks to stack to make one observation
    'include_velocity' : True, # concat dx and dy of particle filter to pose

    'notes' : '229 project'
}

with open('../../weights/{}.yaml'.format(run), 'w+') as f:
    yaml.dump(config, f)

env = PyFEBOLEnv(**config)
obs = env.reset()

nb_episodes = config['nb_episodes']  # number of trajectory rollouts
eps_start = config['eps_start']  # e-greedy threshold start value
eps_end = config['eps_end']  # e-greedy threshold end value
eps_decay = config['eps_decay']  # e-greedy threshold decay
gamma = config['gamma']  # Q-learning discount factor
lr = config['lr']  # NN optimizer learning rate
batch_size = config['batch_size']  # Q-learning batch size
target_update = config['target_update'] # frequency of target network update
memory_size = config['memory_size'] # size of memory for replays
nb_warmup_steps = config['nb_warmup_steps']
use_double_dqn = config['use_double_dqn']
phi_length = config['phi_length']
nb_act_repeat = config['nb_act_repeat']
include_velocity = config['include_velocity']

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, obs, action, next_obs, reward):
        reward = np.array([reward])
        action = np.array([action])
        self.memory.append(Transition(obs, action, next_obs, reward))
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return batch

    def __len__(self):
        return len(self.memory)

'''
different architectures may be used for training and may be imported here.
'''

class Qnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_fc1 = nn.Linear(7, 512)
        self.obs_act1 = nn.ReLU()
    
        self.adv_fc1 = nn.Linear(512, 256)
        self.adv_act1 = nn.ReLU()
        self.adv_fc2 = nn.Linear(256, env.nb_actions)

        self.val_fc1 = nn.Linear(512, 256)
        self.val_act1 = nn.ReLU()
        self.val_fc2 = nn.Linear(256, 1)

    def forward(self, obs, hidden=None):

        obs = self.obs_act1(self.obs_fc1(obs))

        adv = self.adv_act1(self.adv_fc1(obs))
        adv = self.adv_fc2(adv)

        val = self.val_act1(self.val_fc1(obs))
        val = self.val_fc2(val)

        output = val + adv - adv.mean()

        return output

# critic: from beliefs and poses to actions
policy_net = Qnet().double().to(device)
target_net = Qnet().double().to(device)

hard_update(policy_net, target_net)
target_net.eval()

memory = ReplayMemory(memory_size)
optimizer = optim.Adam(policy_net.parameters(), lr, weight_decay=0)

steps_done = 0

def select_action(state):
    '''
    in actor critic, the actor function takes in a state and returns an
    action to perform.
    '''
    global steps_done
    steps_done += 1
    state = torch.from_numpy(state).to(device)

    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * max(0, steps_done - nb_warmup_steps - 1) / eps_decay)
    writer.add_scalar('epsilon', eps_threshold, global_step=steps_done)

    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state).max(1)[1].item()
            return action
    else:
        return np.random.randint(env.nb_actions)

def optimize_model():
    '''
    the actual reinforcement learning stuff, adapted from:
        https://discuss.pytorch.org/t/correct-way-to-do-backpropagation-through-time/11701/2
        https://github.com/fshamshirdar/pytorch-rdpg/blob/master/rdpg.py
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    '''
    if len(memory) < nb_warmup_steps:
        return

    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))


    # setting up tensors            
    state_batch = torch.from_numpy(np.concatenate(batch.state)).to(device)

    action_batch = torch.from_numpy(np.concatenate(batch.action)).unsqueeze(1).double().to(device)
    reward_batch = torch.from_numpy(np.concatenate(batch.reward)).unsqueeze(1).double().to(device)

    next_state_batch = torch.from_numpy(np.concatenate(batch.next_state)).to(device)

    # print('state_batch: ', state_batch.shape)
    # print('action_batch: ', action_batch.shape)
    # print('reward_batch: ', reward_batch.shape)
    # print('next_state_batch: ', next_state_batch.shape)

    # print('state_batch: ', state_batch)
    # print('action_batch: ', action_batch)
    # print('reward_batch: ', reward_batch)
    # print('next_state_batch: ', next_state_batch)
    # print('diff: ', next_state_batch - state_batch)

    if use_double_dqn:
        next_state_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_state_values = target_net(next_state_batch).gather(1, next_state_actions).detach()
    else:
        next_state_values = target_net(next_state_batch).max(1)[0].unsqueeze(1).detach()

    # print('next_state_values: ', next_state_values.shape)
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch
    # print('expected_state_action_values: ', expected_state_action_values.shape)

    state_action_values = policy_net(state_batch).gather(1, action_batch.long())
    # print('state_action_values: ', state_action_values.shape)

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.detach())

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1) # is gradient clamping applicable to this problem?
    optimizer.step()

    writer.add_scalar('loss', loss.item(), global_step=steps_done)
    writer.add_scalar('avg_q', state_action_values.mean().item(), global_step=steps_done)

    del loss

    soft_update(policy_net, target_net, target_update)


'''
training
--------

here's where we actually train the agent

we save the weights every 500 episodes (which is nb_episodes * max_episode_length steps long)
and also at the very end of training.
'''

print("beginning training...")

for i_episode in range(nb_episodes):
    print("beginning episode {}...".format(i_episode))

    # stacking frames for initial observation
    obs = env.reset()
    obs = obs[np.newaxis, :]

    # logging rewards and other values
    ep_reward = 0.0

    for t in range(env.nb_max_episode_steps):

        # Select and perform an action
        action = select_action(obs)
        next_obs, reward, done, _ = env.step(action) # action is two degrees
        next_obs = next_obs[np.newaxis, :]

        ep_reward += reward

        # Store the transition in memory
        memory.push(obs, action, next_obs, reward)

        # Move to the next state
        obs = next_obs

        # Perform one step of the optimization (on the target networks)
        optimize_model()

    print('mean tracking error, near collision rate: ', env.getStats())
    
    writer.add_scalar('ep reward', ep_reward, global_step=steps_done)
    writer.add_scalar('mem usage', torch.cuda.memory_allocated(device=device), global_step=steps_done)
    # log_weights(policy_net, 'Q', steps_done, writer)

    # if i_episode % target_update == 0:
    #     hard_update(policy_net, target_net)

    if i_episode % 500 == 0 and i_episode != 0:
        torch.save(target_net.state_dict(), "../../weights/{}.ep{}.pt".format(run, i_episode))

torch.save(target_net.state_dict(), "../../weights/{}.pt".format(run))

print('complete')
