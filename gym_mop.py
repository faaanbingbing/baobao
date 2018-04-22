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

from math import pi as PI


def make_env(env_id, seed, rank, log_dir=None, env_args={}):
    def _thunk():
        # args
        debug = env_args.get('debug', 0)
        #
        env = MooEnv(id=rank, debug=debug, )
        env.seed(seed + rank)
        return env
    return _thunk


@np.vectorize
def MinMaxScale(x, x_min, x_max, clip=1):
    xn = (x - x_min)/(x_max - x_min)
    if not clip:
        return xn
    return np.clip(xn, 0, 1)


class MooEnv(object):
    """ Multi-objective optimization
    """
    def __init__(self, id=0, debug=0):
        self.id = id
        self.debug = self.id==0 and 1
        
        #~ self.action_space = spaces.Discrete(6)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,) )
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,) )
        
        self.init_mop()
        
        # debug begin
        self.frame_count = 0
        # debug end
        self._gym_vars()
    
    def init_mop(self):
        _path = os.path.dirname(__file__)
        self.estimator = joblib.load(os.path.join(_path, './model_svr.pkl'))
        
        self.D = 60
        self.Dfinal = 30
        
        self.fVc = None
        self.fF = None
        self.fAp = None
        
        self.rVc = None
        self.rF = None
        self.rAp = None
        
        self.LimitVc = 14, 240
        self.LimitF = 0.06, 0.3246
        self.LimitApr = 1.0, 2.75
        self.LimitApf = 0.5, 1.0
        self.Limit6 = zip(self.LimitVc, self.LimitF, self.LimitApf, self.LimitVc, self.LimitF, self.LimitApr)
        self.Limit6 = list(map(np.array, self.Limit6))
        self.LimitPcut = 1e-8, 4000.
        
        self.Vminmax = 14, 247
        self.Fminmax = 0.06, 0.3246
        self.APminmax = 0.5, 2.75
        self.VFAPminmax = list(zip(self.Vminmax, self.Fminmax, self.APminmax))
        
        self.solution_info = None
        self.best_reward = None
    
    def make_obs(self):
        return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp])
    
    def init_states(self, ):
        self.fVc = np.random.uniform(*self.LimitVc)
        self.fF = np.random.uniform(*self.LimitF)
        self.fAp = np.random.uniform(*self.LimitApf)
        
        self.rVc = np.random.uniform(*self.LimitVc)
        self.rF = np.random.uniform(*self.LimitF)
        eps = np.finfo(np.float32).eps
        self.rAp = np.random.uniform(self.LimitApr[0], self.LimitApr[1] - self.fAp - eps)
    
    def check_states(self, obs):
        if (self.Limit6[0]>obs).any() or (self.Limit6[1]<obs).any():
            print('!check_states failed!')
            return False
        return True
    
    def planning_cuts(self, eps=.05):
        """ k: 初始余量 (mm)
        """
        k = (self.D - self.Dfinal)/2
        rAp0, fAp = self.rAp, self.fAp
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
    
    def Pcut(self, Nr):
        """ x_fix: al,40cr,45,80,55,steel
        """
        Xs, x_fix = [], [0,1,0,0,1,0]
        rX = MinMaxScale([self.rVc, self.rF, self.rAp], *self.VFAPminmax)
        fX = MinMaxScale([self.fVc, self.fF, self.fAp], *self.VFAPminmax)
        rX = np.tile(np.append(rX, x_fix), (Nr,1))
        fX = np.append(fX, x_fix)
        Xs = np.vstack((rX, fX))
        return self.estimator.predict(Xs).clip(*self.LimitPcut)
    
    def f_objectives(self, cuts, Lair=15, Lcut=100, Pst=945, t_st=50, t_pct=300):
        """ 注意D是每一刀开始的直径,所以d_f是self.D减去粗加工的剩余
        """
        Ds = [self.D-ap*2 for ap in cuts[:-1]]
        Dr, d_f = [self.D]+Ds[:-1], Ds[-1]
        
        t = lambda D, L, Vc, f: 3.14*D*L/(1000.*Vc*f)
        
        t_air_r = [t(d_r, Lair, self.rVc, self.rF) for d_r in Dr]
        t_air_f = t(d_f, Lair, self.fVc, self.fF)
        t_cut_r = [t(d_r, Lcut, self.rVc, self.rF) for d_r in Dr]
        t_cut_f = t(d_f, Lcut, self.fVc, self.fF)
        
        def Est():
            return Pst*(t_st + sum(t_air_r) + t_air_f + sum(t_cut_r) + t_cut_f)
        
        def Eu():
            pu = lambda Vc,D: -39.45*(1e3*Vc/(PI*D)) + 0.03125*(1e3*Vc/(PI*D))**2 + 17183
            Pu_r = np.array( [pu(self.rVc, d_r) for d_r in Dr] )
            Pu_f = pu(self.fVc, d_f)
            return Pu_r.dot(t_air_r) + Pu_f*t_air_f + Pu_r.dot(t_cut_r) + Pu_f*t_cut_f
        
        def Emr():
            Pmr = self.Pcut(Nr=len(cuts)-1)
            return Pmr[:-1].dot(t_cut_r) + Pmr[-1]*t_cut_f
        
        def Eauc(Pcf=80, Phe=1000):
            return (Pcf+Phe)*(sum(t_air_r) + t_air_f + sum(t_cut_r) + t_cut_f)
        
        T = lambda Vc, f, Ap: 60* 4.43*10**12/(Vc**6.8*f**1.37*Ap**0.24)
        Tr = [1/T(self.rVc, self.rF, ap_r) for ap_r in cuts[:-1]]
        Tf = 1/T(self.fVc, self.fF, cuts[-1])
        t_ect = np.array(t_cut_r).dot(Tr) + t_cut_f*Tf
        
        def Ect():
            return Pst*t_pct*t_ect
        
        SEC = (Est()+Eu()+Emr()+Eauc()+Ect())/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        Tp = t_st + sum(t_air_r) + sum(t_cut_r) + t_air_f + t_cut_f + t_pct*t_ect
        
        def Cp(k0=0.3, ke=0.13, Ch=82.5, ne=2, Ci=2.5):
            kt = Ch/400 + Ci/(0.75*ne)
            return k0*Tp + ke*Tp + kt*t_ect
        
        return SEC, Tp, Cp()
    
    def reward(self, ):
        """ Normalized reward """
        cuts = self.planning_cuts()
        f0, f1, f2 =objs= self.f_objectives(cuts)
        R = -sum(np.tanh(objs)*0.3)
        return R, cuts
    
    def on_step(self, action):
        """ States 
        """
        d_fVc, d_fF, d_fAp, d_rVc, d_rF, d_rAp = np.clip(action, *self.Limit6)
        
        self.fVc += d_fVc
        self.fF += d_fF
        self.fAp += d_fAp
        
        self.rVc += d_rVc
        self.rF += d_rF
        self.rAp += d_rAp
        
        self.fVc, self.fF, self.fAp, \
        self.rVc, self.rF, self.rAp = self.make_obs().clip(*self.Limit6)
    
    def reset(self, ):
        self.init_states()
        self.frame_count = 0
        return self.make_obs()
    
    
    def step(self, action):
        
        info, done = {}, 0
        
        self.on_step(action)
        
        obs = self.make_obs()
        
        if self.check_states(obs) is False:
            done = 1
            reward = -1
            print("Fatal Error: should not happen")
        else:
            reward, cuts = self.reward()
        
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



