'''
util.py
-------

has some utility functions for training and testing
'''
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tensorize_exp(obs, action, next_obs, reward):
    '''
    converts a piece of experience into a torch tensor
    '''
    belief, pose = obs
    belief = torch.tensor(belief).to(device)
    pose = torch.tensor(pose).to(device)
    obs = belief, pose

    if next_obs is not None:
        next_belief, next_pose = next_obs
        next_belief = torch.tensor(next_belief).to(device)
        next_pose = torch.tensor(next_pose).to(device)
        next_obs = next_belief, next_pose
    
    # action = torch.tensor(action).to(device) # action is supposed to be already tensor
    reward = torch.tensor([reward]).to(device)

    return obs, action, next_obs, reward

def tensorize_state(state):
    '''
    states are often bundled as tuples, so you can't just make them into tensors
    '''
    belief, pose = state
    if isinstance(belief, torch.Tensor): return state
    belief = torch.tensor(belief).to(device).double()
    pose = torch.tensor(pose).to(device).double()
    return belief, pose

def add_batch_to_state(state, to_tensor=False):
    '''
    states received from the environment don't have the batch dimension added,
    so we add them here. we also ensure that the states are torch tensors, if
    necessary.
    '''
    if state is None: return None
    if to_tensor:
        belief, pose = tensorize_state(state)
    else:
        belief, pose = state
    if isinstance(belief, torch.Tensor):
        belief = belief.unsqueeze(0)
        pose = pose.unsqueeze(0)
    else:
        belief = belief[np.newaxis, ...]
        pose = pose[np.newaxis, ...]
    return (belief, pose)

class FilterBuffer(object):
    '''
    with just one frame of input, an agent has no way to tell which direction
    the cloud of particles is moving. thus, we stack a number of frames together
    and use the stacked filter as input, giving the agent a chance to estimate
    the target drone's velocity.
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def push(self, filter_):
        '''
        when we're right at the beginning of an episode, there isn't a history
        of filters to stack. we just duplicate the first filter four times to
        seed the buffer.
        '''
        self.buffer.append(filter_)
        while len(self.buffer) < self.capacity:
            self.buffer.append(filter_)
        if len(self.buffer) > self.capacity:
            del self.buffer[0]

    def get(self):
        stacked = np.concatenate(self.buffer, 0)
        return stacked

    def __del__(self):
        while len(self.buffer) > 0:
            del self.buffer[0]

class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.15, 0.01, 0.2
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

def soft_update(net, net_target, target_update=1e-3):
    for param_target, param in zip(net_target.parameters(), net.parameters()):
        param_target.data.copy_(param_target.data * (1.0 - target_update) + param.data * target_update)

def hard_update(net, net_target):
    net_target.load_state_dict(net.state_dict())

def log_weights(model, which, steps_done, writer):
    params = model.state_dict()
    for layer, weights in params.items():
        writer.add_histogram('{}.{}'.format(which, layer), weights, global_step=steps_done)
