import gym
from gym import spaces
from gym.utils import seeding

import torch
import math
import numpy as np
import copy
import collections as col
import os
import time
import random
import socket
import struct

from sklearn import svm
from sklearn.externals import joblib


def make_env(env_id, seed, rank, log_dir=None, env_args={}):
    def _thunk():
        # args
        debug = env_args.get('debug', 0)
        #
        env = MooEnv(id=rank, debug=debug, )
        env.seed(seed + rank)
        return env
    return _thunk


class MooEnv(object):
    """ Multi-objective optimization
    """
    def __init__(self, id=0, debug=0):
        self.id = id
        self.debug = 0
        
        #~ self.action_space = spaces.Discrete(6)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,) )
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,) )
        
        _path = os.path.dirname(__file__)
        self.estimator = joblib.load(os.path.join(_path, './model_svr.pkl'))
        
        self.D = 1.
        self.T = None
        
        self.fVc = None
        self.fF = None
        self.fAp = None
        
        self.rVc = None
        self.rF = None
        self.rAp = None
        
        self.LimitVc = 1e-3, 1.
        self.LimitF = 1e-3, 1.
        self.LimitAp = 1e-3, .5
        
        self.act_min = np.zeros(self.action_space.shape)
        self.act_max = np.zeros(self.action_space.shape)
        
        # debug begin
        self.frame_count = 0
        # debug end
        self._gym_vars()
    
    def gen_obs(self):
        return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp])
    
    def gen_states(self, D=None):
        self.T = self.D if D is None else D
        
        self.fVc = np.random.uniform(*self.LimitVc)
        self.fF = np.random.uniform(*self.LimitF)
        self.fAp = np.random.uniform(*self.LimitAp)
        
        self.rVc = np.random.uniform(*self.LimitVc)
        self.rF = np.random.uniform(*self.LimitF)
        self.rAp = np.random.uniform(*self.LimitAp)
        
        return self.gen_obs()
    
    def check_states(self):
        if self.LimitVc[0]<=self.rVc<=self.LimitVc[1] and \
                    self.LimitF[0]<=self.rF<=self.LimitF[1] and \
                    self.LimitAp[0]<=self.rAp<=self.LimitAp[1] and \
                    self.LimitVc[0]<=self.fVc<=self.LimitVc[1] and \
                    self.LimitF[0]<=self.fF<=self.LimitF[1] and \
                    self.LimitAp[0]<=self.fAp<=self.LimitAp[1]:
            return True
        return False
    
    def gen_cuts(self, rAp0=None, fAp=None, k=1., eps=.05):
        if rAp0 is None:
            rAp0, fAp = np.random.uniform(*self.LimitAp, size=(2,))
        rApL = (k - fAp)%rAp0
        d = (rApL - rAp0) / rAp0
        if 0<d<=eps:
            rApL += rAp0
            Nr = (k - fAp) // rAp0
        else:
            rApL = (k - fAp) % rAp0
            Nr = (k - fAp) // rAp0 + 1
        cuts = [rAp0,]*int(Nr-1) + [rApL,] + [fAp,]
        return cuts
    
    def f_energy(self, cuts):
        """ objective function """
        energy = 0.
        T = self.T
        def get_energy(vc, cd, ap, D):
            X = [vc, cd, ap, 1,0,0,0,1,0]
            energy_ = self.estimator.predict(np.array([X])) * D**2
            return  energy_[0]
        # roughing cut
        for d in cuts[:-1]:
            energy += get_energy(self.rVc, self.rF, self.rAp, T)
            T -= d
        # finishing cut
        energy += get_energy(self.fVc, self.fF, self.fAp, T)
        return energy
    
    def reward(self, cuts):
        """ Normalized reward """
        #~ R = -self.f_energy(cuts)/4000.
        R = -self.f_energy(cuts)
        return R
    
    def make_obs(self, d_fVc, d_fF, d_fAp, d_rVc, d_rF, d_rAp):
        """ States 
        """
        self.fVc += d_fVc
        self.fF += d_fF
        self.fAp += d_fAp
        
        self.rVc += d_rVc
        self.rF += d_rF
        self.rAp += d_rAp
        
        return self.gen_obs()
    
    def reset(self, ):
        obs = self.gen_states()
        
        self.frame_count = 0
        # debug begin
        if self.debug and self.id==0:
            print('reset obs', obs)
        # debug end
        return obs
    
    
    def step(self, action):
        assert self.T is not None
        
        info, done = {}, 0
        
        #~ self.act_min = min(self.act_min, action)
        #~ self.act_max = max(self.act_max, action)
        
        obs = self.make_obs(*action)
        
        if self.check_states() is False:
            done = 1
            reward = -1
        else:
            cuts = self.gen_cuts(self.rAp, self.fAp)
            reward = self.reward(cuts)
        
        self.frame_count += 1
        
        # debug begin
        #~ if self.debug and self.id==0:
            #~ print('cuts', cuts)
            #~ print('action', action)
            #~ print('done', done, reward)
        #~ if self.frame_count>=300:
            #~ self.frame_count = 0
            #~ done = 1
        # debug end
        
        return obs, float(reward), done, info
    
    
    # ----- Rainbow Dqn Special -----
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
    
    def action_space_n(self):
        return self.action_space.n
    
    def close(self):
        pass
    
    # ----- Gym Special -----
    def _gym_vars(self):
        self.seed()
        self._spec = None
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        self.reward_range = (-100.0, 100.0)
        self.repeat_action = 0
    
    @property
    def spec(self):
        return self._spec

    @property
    def unwrapped(self):
        return self
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def render(self, mode='human', close=False):
        """ show something
        """
        return None



