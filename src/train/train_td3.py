'''
Cedrick Argueta

mostly inspired by:
https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
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
import yaml
import os
import matplotlib
import matplotlib.pyplot as plt

import sys
sys.path.append('..')

from PyFEBOLEnv import PyFEBOLEnv
from util import soft_update

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

# naming scheme: date.sensor_modality.test_number.model_architecture
run = 'td3_run_03'

writer = SummaryWriter('../../logs/{}'.format(run))

config = {
    'run' : run,

    # search domain config
    'domain_length' : 200,
    'target_max_step' : 1.7,
    'nb_directions' : 12, # move in 30 degree radially spaced
    'nb_max_episode_steps' : 100,
    'nb_act_repeat' : 1,
    'action_dimensionality' : 'continuous',

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
    'nb_episodes' : int(1e6), # nb_episodes * nb_max_episode_steps / nb_act_repeat = total steps
    'eps_start' : 0.9,
    'eps_end' : 0.1,
    # this is nb_episodes * max_ep_length /nb_act_repeat / 6
    'eps_decay' : int(1e5) / 6, # this number times six is where we reach about 0.1 in steps
    'gamma' : 0.995, # 1 / (1 - gamma) is timescale?
    'lr_mu' : 1e-3,
    'lr_Q' : 1e-3,
    'batch_size' : 128,
    'target_update' : 1e-3, # soft updates
    'memory_size' : int(1e5),
    'nb_warmup_steps' : 10000,
    'phi_length' : 1, # how many ticks to stack to make one observation
    'include_velocity' : True, # concat dx and dy of particle filter to pose
    'reward_scaling' : 1, 
    'update_delay' : 2,

    'notes' : 'TD3 algo'
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
lr_mu = config['lr_mu']
lr_Q = config['lr_Q']
batch_size = config['batch_size']  # Q-learning batch size
target_update = config['target_update'] # frequency of target network update
memory_size = config['memory_size'] # size of memory for replays
nb_warmup_steps = config['nb_warmup_steps']
phi_length = config['phi_length']
nb_act_repeat = config['nb_act_repeat']
include_velocity = config['include_velocity']
reward_scaling = config['reward_scaling']
update_delay = config['update_delay']

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

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        self.q1_fc1 = nn.Linear(7 + 1, 512)
        self.q1_act1 = nn.ReLU()
        self.q1_fc2 = nn.Linear(512, 384)
        self.q1_act2 = nn.ReLU()
        self.q1_fc3 = nn.Linear(384, 1)

        self.q2_fc1 = nn.Linear(7 + 1, 512)
        self.q2_act1 = nn.ReLU()
        self.q2_fc2 = nn.Linear(512, 384)
        self.q2_act2 = nn.ReLU()
        self.q2_fc3 = nn.Linear(384, 1)

    def forward(self, obs, action):

        obs = torch.cat([obs, action], 1)

        out1 = self.q1_act1(self.q1_fc1(obs))
        out1 = self.q1_act2(self.q1_fc2(out1))
        out1 = self.q1_fc3(out1)

        out2 = self.q2_act1(self.q2_fc1(obs))
        out2 = self.q2_act2(self.q2_fc2(out2))
        out2 = self.q2_fc3(out2)

        return out1, out2

    def q1(self, obs, action): 
        obs = torch.cat([obs, action], 1)

        out1 = self.q1_act1(self.q1_fc1(obs))
        out1 = self.q1_act2(self.q1_fc2(out1))
        out1 = self.q1_fc3(out1)

        return out1

class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        self.obs_fc1 = nn.Linear(7, 512)
        self.obs_act1 = nn.ReLU()
    
        self.obs_fc2 = nn.Linear(512, 384)
        self.obs_act2 = nn.ReLU()

        self.obs_fc3 = nn.Linear(384, 1)
        self.obs_act3 = nn.Tanh()

    def forward(self, obs, hidden=None):

        obs = self.obs_act1(self.obs_fc1(obs))
        obs = self.obs_act2(self.obs_fc2(obs))
        actions = self.obs_act3(self.obs_fc3(obs))

        return actions


# actor: from beliefs and poses to actions
mu = Actor().double().to(device)
mu_target = Actor().double().to(device)

# critic: from beliefs and poses and actions to Q-values
Q = Critic().double().to(device)
Q_target = Critic().double().to(device)

mu_target.load_state_dict(mu.state_dict())
Q_target.load_state_dict(Q.state_dict())
mu_target.eval()
Q_target.eval()

memory = ReplayMemory(memory_size)
mu_optim = optim.Adam(mu.parameters(), lr_mu)
Q_optim = optim.Adam(Q.parameters(), lr_Q)

steps_done = 0

def select_action(state):
    '''
    in actor critic, the actor function takes in a state and returns an
    action to perform.
    '''
    global steps_done

    eps = eps_end + (eps_start - eps_end) * math.exp(-1. * max(0, steps_done - nb_warmup_steps - 1) / eps_decay)
    writer.add_scalar('epsilon', eps, global_step=steps_done)

    steps_done += 1

    with torch.no_grad():
        action = mu(torch.from_numpy(state).to(device))
        action = action.cpu().numpy()
        action += eps * max(min(np.random.randn(), 0.5), -0.5)
        action = action.clip(-1, 1)
        action *= np.pi

        assert action >= -np.pi and action <= np.pi, 'action out of bounds: {}'.format(action)
        return action.item()

def optimize_model(t):
    '''
    the actual reinforcement learning stuff, adapted from:
        https://discuss.pytorch.org/t/correct-way-to-do-backpropagation-through-time/11701/2
        https://github.com/fshamshirdar/pytorch-rdpg/blob/master/rdpg.py
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
    '''
    if len(memory) < nb_warmup_steps and len(memory) != memory_size:
        return

    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    # setting up tensors            
    state_batch = torch.from_numpy(np.concatenate(batch.state)).to(device)

    action_batch = torch.from_numpy(np.stack(batch.action)).to(device)
    reward_batch = torch.from_numpy(np.stack(batch.reward)).to(device).double()
   
    non_final_next_state_batch = torch.from_numpy(np.concatenate(batch.next_state)).to(device)

    # print('shapes!')
    # print('state_batch: ', state_batch.shape)
    # print('action_batch: ', action_batch.shape)
    # print('reward_batch: ', reward_batch.shape)
    # print('non_final_next_state_batch: ', non_final_next_state_batch.shape)

    with torch.no_grad(): # no grad because these are target networks
        target_actions = mu_target(non_final_next_state_batch)
        # should i add noise here like in TD3?
        noise = torch.from_numpy(np.clip(np.random.randn(*target_actions.shape) * 0.2, -0.5, 0.5))
        next_state_values1, next_state_values2  = Q_target(non_final_next_state_batch, target_actions + noise)
        next_state_values = torch.min(next_state_values1, next_state_values2)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # critic update
    Q_optim.zero_grad()

    state_action_values1, state_action_values2 = Q(state_batch, action_batch)

    Q_loss = F.smooth_l1_loss(state_action_values1, expected_state_action_values.detach()) \
        + F.smooth_l1_loss(state_action_values2, expected_state_action_values.detach())
    writer.add_scalar('Q_loss', Q_loss.item(), global_step=steps_done)

    Q_loss.backward()
    Q_optim.step()

    del Q_loss

    if t % update_delay == 0:
        # actor update
        mu_optim.zero_grad()

        # mu_loss and state_action_values should be nearly identical since we
        # are using the same Q function for both, and actions were selected
        # with the same policy.
        mu_loss = -Q.q1(state_batch, mu(state_batch)).mean()
        writer.add_scalar('mu_loss', mu_loss.item(), global_step=steps_done)

        mu_loss.backward()
        mu_optim.step()

        del mu_loss

        soft_update(mu, mu_target)
        soft_update(Q, Q_target)

def soft_update(net, net_target):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - target_update) + param.data * target_update)

'''
training
--------

here's where we actually train the agent. a new filter buffer is created for 
every episode to stack the observations for us. in practice, we shape the 
reward function by squaring it, penalizing outliers more (see L2 loss for more).

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
    logging = (i_episode % 100 == 0)
    ep_reward = 0.0

    for t in range(env.nb_max_episode_steps):

        # Select and perform an action
        action = select_action(obs)
        action_tuple = (action, 0) # heading angle is ineffectual for bearing only sensor
        next_obs, reward, done, _ = env.step(action_tuple) # action is two degrees
        next_obs = next_obs[np.newaxis, :]

        # some papers report that reward scaling helps -- hopefully it does!
        reward *= reward_scaling

        ep_reward += reward

        # Store the transition in memory
        memory.push(obs, action, next_obs, reward)

        # Move to the next state
        obs = next_obs

        # Perform one step of the optimization (on the target networks)
        optimize_model(t)

    print('mean tracking error, near collision rate: ', env.getStats())

    writer.add_scalar('ep reward', (1. / reward_scaling) * ep_reward, global_step=steps_done)
    writer.add_scalar('mem usage', torch.cuda.memory_allocated(device=device), global_step=steps_done)
    # log_weights(Q, 'Q', steps_done, writer)
    # log_weights(mu, 'mu', steps_done, writer)

    if i_episode % 500 == 0 and i_episode != 0:
        torch.save(Q_target.state_dict(), "../../weights/{}.ep{}.Q.pt".format(run, i_episode))
        torch.save(mu_target.state_dict(), "../../weights/{}.ep{}.mu.pt".format(run, i_episode))

torch.save(Q_target.state_dict(), "../../weights/{}.Q.pt".format(run))
torch.save(mu_target.state_dict(), "../../weights/{}.mu.pt".format(run))

print('complete')
