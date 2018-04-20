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
        self.debug = self.id==0 and 1
        
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
        
        self.LimitVc = 1e-8, 1.
        self.LimitF = 1e-8, 1.
        self.LimitAp = 1e-8, .4
        self.Limits = zip(self.LimitVc, self.LimitF, self.LimitAp)
        self.Limitx = zip(self.LimitVc, self.LimitF, self.LimitAp, self.LimitVc, self.LimitF, self.LimitAp)
        self.Limits = list(map(np.array, self.Limits))
        self.Limitx = list(map(np.array, self.Limitx))
        self.LimitEnergy = 1e-8, 4000.
        
        self.solution_info = None
        self.best_reward = None
        
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
        #~ if self.LimitVc[0]<=self.rVc<=self.LimitVc[1] and \
                    #~ self.LimitF[0]<=self.rF<=self.LimitF[1] and \
                    #~ self.LimitAp[0]<=self.rAp<=self.LimitAp[1] and \
                    #~ self.LimitVc[0]<=self.fVc<=self.LimitVc[1] and \
                    #~ self.LimitF[0]<=self.fF<=self.LimitF[1] and \
                    #~ self.LimitAp[0]<=self.fAp<=self.LimitAp[1]:
            #~ return True
        #~ return False
        check_list = [self.rVc, self.rF, self.rAp, self.fVc, self.fF, self.fAp]
        if (self.Limitx[0]>check_list).any() or (self.Limitx[1]<check_list).any():
            print('!check_states failed!')
            return False
        return True
    
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
        cuts = [rAp0,]*int(Nr-1) + [rApL, fAp]
        return cuts
    
    def f_energy(self, cuts):
        """ objective function
        @return: energy * D^2
        """
        energy = 0.
        T = self.T
        Ts = []
        Xs, x_fix = [], [1,0,0,0,1,0]
        def get_energy(X):
            return  self.estimator.predict(X)
        # roughing cut
        for d in cuts[:-1]:
            Xs.append([self.rVc, self.rF, self.rAp] + x_fix)
            Ts.append(T)
            T -= d
        # roughing end
        assert 0<T<1
        # finishing cut
        Xs.append([self.fVc, self.fF, self.fAp] + x_fix)
        Ts.append(T)
        # finishing end
        D = np.array(Ts)
        Xs = np.array(Xs)
        energy = get_energy(Xs)
        energy = energy.clip(*self.LimitEnergy) / self.LimitEnergy[1]
        energy *= D**2
        # debug begin
        #~ if self.debug:
            #~ print('D %r E %r'%(D.shape, energy.shape))
        # debug end
        return energy.sum()
    
    def reward(self, cuts):
        """ Normalized reward """
        R = -self.f_energy(cuts)
        return R
    
    def make_obs(self, d_fVc, d_fF, d_fAp, d_rVc, d_rF, d_rAp):
        """ States 
        """
        if 1:
            d_fVc = np.clip(d_fVc, *self.LimitVc)
            d_fF = np.clip(d_fF, *self.LimitF)
            d_fAp = np.clip(d_fAp, *self.LimitAp)
            
            d_rVc = np.clip(d_rVc, *self.LimitVc)
            d_rF = np.clip(d_rF, *self.LimitF)
            d_rAp = np.clip(d_rAp, *self.LimitAp)
        
        self.fVc += d_fVc
        self.fF += d_fF
        self.fAp += d_fAp
        
        self.rVc += d_rVc
        self.rF += d_rF
        self.rAp += d_rAp
        
        if 1:
            self.fVc = np.clip(self.fVc, *self.LimitVc)
            self.fF = np.clip(self.fF, *self.LimitF)
            self.fAp = np.clip(self.fAp, *self.LimitAp)
            
            self.rVc = np.clip(self.rVc, *self.LimitVc)
            self.rF = np.clip(self.rF, *self.LimitF)
            self.rAp = np.clip(self.rAp, *self.LimitAp)
        
        return self.gen_obs()
    
    def reset(self, ):
        obs = self.gen_states()
        
        self.frame_count = 0
        # debug begin
        #~ if self.debug:
            #~ print('reset obs', obs)
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
        
        # limit steps begin
        self.frame_count += 1
        if self.frame_count>=1e3:
            self.frame_count = 0
            done = 1
        # limit steps end
        
        # debug begin
        if self.solution_info is None or reward>self.solution_info[1]:
            solution = 'solution Nc:%d r(Vc%.3f F%.3f Ap%.3f) f(Vc%.3f F%.3f Ap%.3f) r:%g'%\
                        (len(cuts), self.rVc, self.rF, self.rAp, self.fVc, self.fF, self.fAp, reward)
            self.solution_info = [solution, reward, 1]
            if self.best_reward is None:
                self.best_reward = reward
        #~ if self.debug:
            #~ print('cuts %r %r reward %r'%(sum(cuts), len(cuts), reward))
        if done:
            #~ if self.debug:
                #~ print('cuts %r %r reward %r'%(sum(cuts), len(cuts), reward))
                #~ print(self.solution_info)
            if self.best_reward<self.solution_info[1]:
                info.update(solution=self.solution_info)
                self.best_reward = self.solution_info[1]
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



