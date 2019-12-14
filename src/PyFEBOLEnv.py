'''
Cedrick Argueta

glue between PyFEBOL and deep-drone-localization
'''

import random
import math
import numpy as np

import gym
from gym.utils import seeding
from gym import error, spaces, utils

import PyFEBOL
from PyFEBOL import drone
from PyFEBOL import sensor
from PyFEBOL import searchdomain
from PyFEBOL import filter
from PyFEBOL import policy
from PyFEBOL import cost
from PyFEBOL import util

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import os

plt.ion()

class PyFEBOLEnv(gym.Env):
    '''
    this class turns the PyFEBOL simulation environment into a gym environment.

    configuration parameters are passed as arguments, however sensible defaults
    are given.
    '''
    def __init__(self, **kwargs):
        '''
        gathers the configuration parameters from arguments and sets defaults
        as necessary.
        '''
        # search domain config 
        self.target_max_step = kwargs.get('target_max_step', 1.7)
        self.domain_length = kwargs.get('domain_length', 100)
        self.nb_max_episode_steps = kwargs.get('nb_max_episode_steps', 100)
        self.nb_act_repeat = kwargs.get('nb_act_repeat', 1)
        # repeating actions makes it so that we traverse the same area in
        # a fewer amount of total steps, so to make comparison equal when
        # we repeat actions we scale the episode length
        self.nb_max_episode_steps = int(self.nb_max_episode_steps / self.nb_act_repeat)

        # cost model config
        self.lambda_1 = kwargs.get('lambda_1', 1.)
        self.lambda_2 = kwargs.get('lambda_2', 1.)
        self.lambda_3 = kwargs.get('lambda_3', 1.)
        self.entropy_threshold = kwargs.get('entropy_threshold', 0.5)
        self.distance_threshold = kwargs.get('distance_threshold', 15.0)
        self.tracking_threshold = kwargs.get('tracking_threshold', 0.85)

        # filter config
        self.filter_length = kwargs.get('filter_length', 50)
        self.nb_particles = kwargs.get('nb_particles', 1000)
        self.include_velocity = kwargs.get('include_velocity', False)

        # sensor config
        self.sensor_type = kwargs.get('sensor_type', 'bearing_only')
        if self.sensor_type == 'bearing_only':
            self.sensor_sd = kwargs.get('sensor_sd', 10.0)
        
        # setting up action space to be discrete or continuous
        self.action_dimensionality = kwargs.get('action_dimensionality', 'discrete')
        if self.action_dimensionality == 'discrete':
            self.nb_directions = kwargs.get('nb_directions', 36)
            self.headings = kwargs.get('headings', None)

        # setting up stat trak
        self.total_tracking_error = 0
        self.total_near_collisions = 0

        self.seed()
        self.viewer = None
        self.num_steps = 0
        self.seeker_hist = []


    def _setup(self):
        '''
        creates the PyFEBOL simulation objects as specified by the config params
        '''
        choices = { (0.1, 0.1) : [(self.target_max_step, 0), (0, self.target_max_step)], (0.9, 0.1) : [(-self.target_max_step, 0), (0, self.target_max_step)], (0.1, 0.9) : [(self.target_max_step, 0), (0, -self.target_max_step)] , (0.9, 0.9) : [(-self.target_max_step, 0), (0, -self.target_max_step)] }
        self.init = random.choice(list(choices))
        dx, dy = random.choice(choices[self.init])
        self.target_p = policy.ConstantVelocityPolicy(dx, dy, maxStep=self.target_max_step)
        self.m = searchdomain.SearchDomain(self.domain_length, policy=self.target_p, init=self.init)

        if self.sensor_type == 'bearing_only':
            self.s = sensor.BearingOnlySensor(self.sensor_sd)

        # maxStep 2, target moves at 1.7
        self.f = filter.ParticleFilter(self.m, self.filter_length, self.s, 2.0, self.nb_particles)

        self.d = drone.Drone(self.domain_length / 2, self.domain_length / 2, 60, 5.0, 15.0, self.s, self.m) # start in center of domain, facing 60 degrees, 5.0 maxStep, 15.0 degrees headingMaxStep

        if self.action_dimensionality == 'discrete':
            self.p = policy.RLPolicy(self.d.maxStep, self.nb_directions, headings=self.headings)
            self.action_space = self.p.actions
            self.nb_actions = len(self.action_space)
        elif self.action_dimensionality == 'continuous':
            pass # nothing to do

        self.c = cost.WeightedThresholdCostModel(self.distance_threshold, self.entropy_threshold, self.tracking_threshold, self.lambda_1, self.lambda_2, self.lambda_3)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        performs one step of simulation: takes an action, moves the pieces around,
        updates the internals of the simulator, and spits out the observation
        and reward.
        '''
        self.num_steps += 1

        if self.action_dimensionality == 'discrete':
            # in this case, the action passed in originally is the index of the
            # action we should take
            action = self.action_space[action]
        elif self.action_dimensionality == 'continuous':
            # in this case, the action passed is a list of the two controls we
            # wish to take
            move_angle, heading_angle = action
            ax = self.d.maxStep * np.cos(move_angle) 
            ay = self.d.maxStep * np.sin(move_angle) 
            action = (ax, ay, heading_angle) # remember that heading must be between 1, -1
        self.d.act(action, self.nb_act_repeat)

        self.m.moveTarget(self.nb_act_repeat)

        obs = self.s.observe(self.m.getTheta(), self.d.getPose())
        self.f.update(self.d.getPose(), obs, self.nb_act_repeat)

        if self.include_velocity:
            next_state = np.concatenate([np.array(self.d.getPose()), np.array(self.f.mean_velocity()), np.array(self.f.centroid())]) # returning a tuple of np arrays
        else:
            next_state = np.concatenate([np.array(self.d.getPose()), np.array(self.f.centroid())]) # returning a tuple of np arrays

        reward = self.c.getCost(self.m, self.d, self.f, action)

        done = self._isDone()

        if self.num_steps > self.nb_max_episode_steps / 4:
            self.total_tracking_error += np.linalg.norm(np.array(self.f.centroid()) - np.array([self.m.getTheta()]))
        x, y, _ = self.d.getPose()
        self.total_near_collisions += int(np.linalg.norm(np.array([x, y]) - np.array([self.m.getTheta()])) < self.distance_threshold)
        
        return next_state, reward, done, {}

    def getStats(self):
        '''
        return mean tracking error and near collision rate for this run
        '''
        return self.total_tracking_error / self.num_steps, self.total_near_collisions / self.num_steps

    def _isDone(self):
        '''
        by only terminating the episode after a max number of steps,
        we ensure that the agent aims to maximize reward over the whole episode
        rather than look for early stopping criteria
        '''
        return self.num_steps > (self.nb_max_episode_steps - 1)

    def render(self, **kwargs):
        '''
        text mode is for printing out information during training
        live mode is meant to be for visualization of the agents performance
            during training, but it isn't tested currently
        save mode is for creating gifs that visualize performance, which are
            saved to disk
        '''
        mode = kwargs.get('mode', 'text')
        freq = kwargs.get('freq', 10) # how often we print
        path = kwargs.get('path', '../gifs/{}.png'.format(self.num_steps))
        title = kwargs.get('title', 'Belief Distribution')
        reward = kwargs.get('reward', 0)
        ep_reward = kwargs.get('ep_reward', 0)

        if mode == 'text':
            if self.num_steps % freq == 0:
                x, y, heading = self.d.getPose()
                theta_x, theta_y = self.m.getTheta()
                print("step_num is: ", self.num_steps)
                if self.init is not None:
                    print('initial theta position: x: {}, y: {}'.format(self.init[0]*self.domain_length, self.init[1]*self.domain_length))
                if isinstance(self.target_p, policy.ConstantVelocityPolicy):
                    print('theta dx: {}, theta dy: {}'.format(self.target_p.dx, self.target_p.dy))
                print('x: {0}, y: {1}'.format(x, y))
                print('theta_x: {0}, theta_y: {1}'.format(theta_x, theta_y))
                print('entropy is: ', self.f.entropy())
                print('max prob is: ', self.f.getBelief().max())
                tracking = np.linalg.norm(np.array(self.f.centroid()) - np.array([theta_x, theta_y]))
                print('tracking error is: ', tracking)
                print('acc. reward is: ', ep_reward)

        elif mode == 'save':
            plt.figure(figsize=(8, 8))
            plt.title('{}, step {}'.format(title, self.num_steps))

            # plotting belief dist
            x_particles = self.f.x_particles
            y_particles = self.f.y_particles
            plt.xlim(0, self.domain_length)
            plt.ylim(0, self.domain_length)
            plt.scatter(x_particles, y_particles, marker='o', c='k', s= (72./plt.gcf().dpi) ** 2) 

            # plot history
            cmap = matplotlib.cm.get_cmap('cividis')
            for i, oldPose in enumerate(self.seeker_hist):
                old_x, old_y, old_heading = oldPose
                marker = matplotlib.markers.MarkerStyle(marker=r'$\wedge$')
                idx = i / len(self.seeker_hist)
                c = cmap(idx)
                plt.scatter(old_x, old_y, marker=marker, color=c)

            # plotting seeker and target drone positions
            x, y, heading = self.d.getPose()
            theta_x, theta_y = self.m.getTheta()
            marker = matplotlib.markers.MarkerStyle(marker=r'$\wedge$')

            plt.scatter(x, y, marker=marker, c='c')
            plt.plot(theta_x, theta_y, 'rx')

            if self.num_steps % 2 == 0:
                self.seeker_hist.append((x, y, heading))

            # important values
            ent = self.f.entropy()
            max_eig = self.f.maxEigenvalue()
            max_prob = self.f.maxProbBucket()
            dist = np.linalg.norm(np.array([x, y]) - np.array([theta_x, theta_y]))
            tracking = np.linalg.norm(np.array(self.f.centroid()) - np.array([theta_x, theta_y]))
            mean_dx, mean_dy = self.f.mean_velocity()
            toptxt = 'Entropy: {:6.2f}, Max Eig: {:8.2f}, Max Prob: {:.3f}, Mean (vx, vy): ({:.1f}, {:.1f})'.format(ent, max_eig, max_prob, mean_dx, mean_dy)
            bottxt = 'Seeker Distance: {:6.2f}, Tracking Error: {:6.2f}, Step Reward: {:4.2f}, Acc. Reward: {:6.2f}'.format(dist, tracking, reward, ep_reward)
            plt.figtext(0.01, 0.01, bottxt, wrap=True, horizontalalignment='left', fontsize=8)
            plt.figtext(0.01, 0.04, toptxt, wrap=True, horizontalalignment='left', fontsize=8)

            # saving graph
            plt.xlim(0, self.m.length)
            plt.ylim(0, self.m.length)
            plt.savefig(path)
            plt.close()

        elif mode == 'live':
            plt.figure(figsize=(8, 8))
            plt.title('{}, step {}'.format(title, self.num_steps))

            # plotting belief dist
            x_particles = self.f.x_particles
            y_particles = self.f.y_particles
            plt.xlim(0, self.domain_length)
            plt.ylim(0, self.domain_length)
            plt.scatter(x_particles, y_particles, marker='o', c='k', s= (72./plt.gcf().dpi) ** 2) 

            # plot history
            cmap = matplotlib.cm.get_cmap('cividis')
            for i, oldPose in enumerate(self.seeker_hist):
                old_x, old_y, old_heading = oldPose
                marker = matplotlib.markers.MarkerStyle(marker=r'$\wedge$')
                idx = i / len(self.seeker_hist)
                c = cmap(idx)
                plt.scatter(old_x, old_y, marker=marker, color=c)

            # plotting seeker and target drone positions
            x, y, heading = self.d.getPose()
            marker = matplotlib.markers.MarkerStyle(marker=r'$\wedge$')

            plt.scatter(x, y, marker=marker, c='c')

            if self.num_steps % 2 == 0:
                self.seeker_hist.append((x, y, heading))

            # important values
            ent = self.f.entropy()
            max_eig = self.f.maxEigenvalue()
            max_prob = self.f.maxProbBucket()
            mean_dx, mean_dy = self.f.mean_velocity()
            toptxt = 'Entropy: {:6.2f}, Max Eig: {:8.2f}, Max Prob: {:.3f}, Mean (vx, vy): ({:.1f}, {:.1f})'.format(ent, max_eig, max_prob, mean_dx, mean_dy)
            plt.figtext(0.01, 0.04, toptxt, wrap=True, horizontalalignment='left', fontsize=8)

            # saving graph
            plt.xlim(0, self.m.length)
            plt.ylim(0, self.m.length)
            plt.draw()

    def close(self):
        pass

    def reset(self):
        self.total_tracking_error = 0
        self.total_near_collisions = 0

        plt.close('all')
        self.num_steps = 0
        self.seeker_hist = []

        self._setup()

        if self.include_velocity:
            obs = np.concatenate([np.array(self.d.getPose()), np.array(self.f.mean_velocity()), np.array(self.f.centroid())]) # returning a tuple of np arrays
        else:
            obs = np.concatenate([np.array(self.d.getPose()), np.array(self.f.centroid())]) # returning a tuple of np arrays
        return obs
