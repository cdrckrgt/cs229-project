'''
Cedrick Argueta

some code taken from:
https://github.com/takuseno/rsvg/blob/master/network.py
https://github.com/fshamshirdar/pytorch-rdpg
https://github.com/seungeunrho/minimalRL
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

from util import tensorize_exp, tensorize_state, add_batch_to_state, \
    FilterBuffer, soft_update, log_weights, OrnsteinUhlenbeckNoise

'''
configuration
-------------

code in this section does multiple things for setting up the experiment.

things to note:
    creating the yaml file here is essential to setting up reproducible experimental
        conditions. training, testing, and baselines depend on this yaml file.
'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("running on: ", device)

run = 'verify_ddpg'

env = gym.envs.make('Pendulum-v0')

env.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

writer = SummaryWriter('../logs/{}'.format(run))

# run hyperparameters
nb_episodes = 100000
nb_max_episode_steps = 500
gamma = 0.99
lr_mu = 1e-4
lr_Q = 1e-3
mu_weight_decay = 0
Q_weight_decay = 0
batch_size = 64
target_update = 1e-3
memory_size = 1000000
nb_warmup_steps = 100
eps_start = 1.0
eps_end = 0
eps_decay = 10000 / 6

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    '''
    a class that holds all the experience for training.
    '''
    def __init__(self, capacity, nb_multi_step=1):
        self.capacity = capacity
        self.memory = []

    def push(self, obs, action, next_obs, reward):
        reward = torch.tensor([reward]).to(device)
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

        self.obs_fc1 = nn.Linear(3, 400)
        self.obs_act1 = nn.ReLU()
        self.obs_fc2 = nn.Linear(400, 300)
        self.obs_act2 = nn.ReLU()
    
        self.merged_fc1 = nn.Linear(301, 300)
        self.merged_act1 = nn.ReLU()
        self.merged_fc2 = nn.Linear(300, 200)
        self.merged_act2 = nn.ReLU()

        self.merged_fc3 = nn.Linear(200, 200)
        self.merged_act3 = nn.ReLU()
        self.merged_fc4 = nn.Linear(200, 1)

    def forward(self, obs, action):

        obs = self.obs_act1(self.obs_fc1(obs))
        obs = self.obs_act2(self.obs_fc2(obs))

        merged = torch.cat([obs, action], 1)
        merged = self.merged_act1(self.merged_fc1(merged))
        merged = self.merged_act2(self.merged_fc2(merged))
        merged = self.merged_act3(self.merged_fc3(merged))
        output = self.merged_fc4(merged)

        return output

class Actor(nn.Module):
    '''
    actor

    takes a state and returns an action

    the network actually outputs the parameters for the action to be taken,
    and the action is sampled from that distribution
    '''
    def __init__(self):
        super().__init__()
    
        self.obs_fc1 = nn.Linear(3, 400)
        self.obs_act1 = nn.ReLU()

        self.obs_fc2 = nn.Linear(400, 300)
        self.obs_act2 = nn.ReLU()

        self.obs_fc3 = nn.Linear(300, 200)
        self.obs_act3 = nn.ReLU()

        self.obs_fc4 = nn.Linear(200, 1)
        self.obs_act4 = nn.Tanh()

    def forward(self, obs):

        obs = self.obs_act1(self.obs_fc1(obs))
        obs = self.obs_act2(self.obs_fc2(obs))
        obs = self.obs_act3(self.obs_fc3(obs))

        action = self.obs_act4(self.obs_fc4(obs))

        return action

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

memory = ReplayMemory(memory_size, nb_max_episode_steps)
mu_optim = optim.Adam(mu.parameters(), lr_mu, weight_decay=mu_weight_decay)
Q_optim = optim.Adam(Q.parameters(), lr_Q, weight_decay=Q_weight_decay)

steps_done = 0

def select_action(state, ou_noise):
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
        noise = eps * ou_noise()[0]
        action =  action.squeeze(0) + noise
        action = action.clip(-2, 2)
        return action

def optimize_model():
    '''
    the actual reinforcement learning stuff, adapted from:
        https://discuss.pytorch.org/t/correct-way-to-do-backpropagation-through-time/11701/2
        https://github.com/fshamshirdar/pytorch-rdpg/blob/master/rdpg.py
        https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py

    modified to support models with multiple inputs (really, it's specific to this
    problem with an image input and vector input).
    '''
    if len(memory) < nb_warmup_steps and len(memory) != memory_size:
        return

    transitions = memory.sample(batch_size)

    batch = Transition(*zip(*transitions))

    # setting up tensors            
    state_batch = torch.from_numpy(np.concatenate(batch.state)).to(device)

    action_batch = torch.from_numpy(np.stack(batch.action)).to(device)
    reward_batch = torch.stack(batch.reward).to(device).double()
   
    non_final_next_state_batch = torch.from_numpy(np.concatenate(batch.next_state)).to(device)

    with torch.no_grad(): # no grad because these are target networks
        target_actions = mu_target(non_final_next_state_batch)
        next_state_values  = Q_target(non_final_next_state_batch, target_actions)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # critic update
    Q_optim.zero_grad()

    state_action_values = Q(state_batch, action_batch)

    Q_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.detach())
    writer.add_scalar('Q_loss', Q_loss.item(), global_step=steps_done)

    Q_loss.backward()
    Q_optim.step()

    del Q_loss

    # actor update
    mu_optim.zero_grad()

    # mu_loss and state_action_values should be nearly identical since we
    # are using the same Q function for both, and actions were selected
    # with the same policy. for RSVG specifically, they will be different
    # because actions are randomly rather than deterministically sampled
    mu_loss = -Q(state_batch, mu(state_batch)).mean()
    writer.add_scalar('mu_loss', mu_loss.item(), global_step=steps_done)

    mu_loss.backward()
    mu_optim.step()

    del mu_loss

    # for m, (name, param) in enumerate(mu.named_parameters()):
    #     if m == 0:
    #         print('name: ', name)
    #         param_scale = np.linalg.norm(param.data.cpu().view(-1))
    #         update = param.grad.data * lr_mu
    #         update_scale = np.linalg.norm(update.cpu().view(-1))
    #         print('param_scale: ', param_scale)
    #         print('update_scale: ', update_scale)
    #         print('ratio: ', update_scale / param_scale)

    soft_update(mu, mu_target, target_update)
    soft_update(Q, Q_target, target_update)

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
    ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    # logging rewards and other values
    logging = (i_episode % 100 == 0)
    ep_reward = 0.0

    for t in range(nb_max_episode_steps):

        # Select and perform an action
        action = select_action(obs, ou_noise)
        next_obs, reward, done, _ = env.step(action) # action is two degrees
        next_obs = next_obs[np.newaxis, :]

        # we want to keep runs comparable between the RL approach and greedy
        # / MCTS solvers, so we scale the rewards to have the same absolute max
        # i suppose this is a fine proxy for the cumulative reward of the 
        # repeated actions
        
        ep_reward += reward

        # Store the transition in memory
        memory.push(obs, action, next_obs, reward)

        # Move to the next state
        obs = next_obs

        # Perform one step of the optimization (on the target networks)
        optimize_model()

    writer.add_scalar('ep reward', ep_reward, global_step=steps_done)
    writer.add_scalar('mem usage', torch.cuda.memory_allocated(device=device), global_step=steps_done)
    log_weights(Q, 'Q', steps_done, writer)
    log_weights(mu, 'mu', steps_done, writer)

    if i_episode % 500 == 0 and i_episode != 0:
        torch.save(Q_target.state_dict(), "../weights/{}.ep{}.Q.pt".format(run, i_episode))
        torch.save(mu_target.state_dict(), "../weights/{}.ep{}.mu.pt".format(run, i_episode))

torch.save(Q_target.state_dict(), "../weights/{}.Q.pt".format(run))
torch.save(mu_target.state_dict(), "../weights/{}.mu.pt".format(run))

print('complete')
