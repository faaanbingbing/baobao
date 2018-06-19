from __future__ import division

import gym
from gym import spaces
from gym.utils import seeding

#~ import torch
import math
import numpy as np
import copy
import collections as col
import os
import time
import random

from sklearn import svm
from sklearn.externals import joblib

import pygmo as pg

from math import pi as PI


def make_env(env_id, seed, rank, log_dir=None, env_args={}):
    def _thunk():
        # args
        debug = env_args.get('debug', 0)
        #
#         env = MooEnv(id=rank, debug=debug, )
        env = MooEnv2(id=rank, debug=debug, )
        # Benchmark Begin
        #~ env = MooEnvX(id=rank, debug=debug, )
        # Benchmark End
        env.seed(seed + rank)
        return env
    return _thunk

def print_list(L):
    for i in L:
        print('\t%.3f'%i)

@np.vectorize
def MinMaxScale(x, x_min, x_max, clip=1):
    xn = (x - x_min)/(x_max - x_min)
    if not clip:
        return xn
    return np.clip(xn, 0, 1)


class MooEnv(object):
    """ Multi-objective optimization
    """
    def __init__(self, id=0, debug=0, obs_type=0):
        self.id = id
        self.debug = self.id==0 and 1
        self.obs_type = obs_type
        #~ self.action_space = spaces.Discrete(6)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,) )
        
        if self.obs_type==0:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,) )
        else:
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,) )
        
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
#         self.Dfinal = 32
        #~ self.Dfinal = 40
        self.K = (self.D - self.Dfinal)/2
        
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
        
        self.Limit3f = zip(self.LimitVc, self.LimitF, self.LimitApf)
        self.Limit3f = list(map(np.array, self.Limit3f))
        self.Limit3r = zip(self.LimitVc, self.LimitF, self.LimitApr)
        self.Limit3r = list(map(np.array, self.Limit3r))
        
        self.Limit6 = zip(self.LimitVc, self.LimitF, self.LimitApf, self.LimitVc, self.LimitF, self.LimitApr)
        self.Limit6 = list(map(np.array, self.Limit6))
        self.LimitPcut = 1e-8, 4000.
        
        self.Vminmax = 14, 247
        self.Fminmax = 0.06, 0.3246
        self.APminmax = 0.5, 2.75
        self.VFAPminmax = list(zip(self.Vminmax, self.Fminmax, self.APminmax))
        
        self.obj_prev = None
        self.solution_info = None
        self.best_reward = None
    
    def make_obs(self, obj=0, randn=0):
        if randn:
           return np.random.randn(self.observation_space.shape[0])
        elif obj:
            return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp, self.obj_prev[0]])
        return np.array([self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp])
    
    def set_states(self, fVc, fF, fAp, rVc, rF, rAp):
        """ debug only """
        self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp = fVc, fF, fAp, rVc, rF, rAp
    
    def init_states(self, ):
        self.fVc = np.random.uniform(*self.LimitVc)
        self.fF = np.random.uniform(*self.LimitF)
        self.fAp = np.random.uniform(*self.LimitApf)
        
        self.rVc = np.random.uniform(*self.LimitVc)
        self.rF = np.random.uniform(*self.LimitF)
        LimitApr = self.LimitApr[1] - self.fAp - np.finfo(np.float32).eps
        assert self.LimitApr[0] <= LimitApr
        self.rAp = np.random.uniform(self.LimitApr[0], LimitApr)
    
    def check_states(self, obs):
        if (self.Limit6[0]>obs).any() or (self.Limit6[1]<obs).any():
            print('!check_states failed!')
            return False
        return True
    
    def planning_cuts(self, eps=.05):
        """ k: 初始余量 (mm)
        """
        k = (self.D - self.Dfinal)/2
        #~ assert k>0
        rAp0, fAp = self.rAp, self.fAp
        rApL = (k - fAp)%rAp0
        #~ assert 0<=rApL and rApL<=rAp0
        d = rApL / rAp0
        if d<=eps:
            rApL += rAp0
            Nr = (k - fAp) // rAp0
        else:
            rApL = (k - fAp) % rAp0
            Nr = (k - fAp) // rAp0 + 1
        cuts = [rAp0,]*int(Nr-1) + [rApL, fAp]
        #~ assert Nr>1 and min(cuts)>0
        # debug begin
        #~ if not (np.array(cuts)>0).all():
            #~ print(len(cuts), min(cuts))
            #~ print(cuts)
        # debug end
        return cuts
    
    def Pcut(self, cuts):
        """ x_fix: al,40cr,45,80,55,steel
        """
        Xs, x_fix = [], [0,1,0,0,1,0]
        for rAp in cuts[:-1]:
            rX = MinMaxScale([self.rVc, self.rF, rAp], *self.VFAPminmax)
            Xs.append( np.append(rX, x_fix) )
        fX = MinMaxScale([self.fVc, self.fF, cuts[-1]], *self.VFAPminmax)
        Xs.append( np.append(fX, x_fix) )
        Xs = np.array(Xs)
        Pc = self.estimator.predict(Xs).clip(*self.LimitPcut)
        # debug begin
        #~ print(Xs.shape, Xs)
        #~ print('Pc', Pc.shape)
        #~ for p in Pc:
            #~ print('\t%.3f'%p)
        # debug end
        return Pc
    
    def f_objectives(self, cuts, Lair=15, Lcut=100, Pst=945, t_st=50, t_pct=300):
        """ 注意D是每一刀开始的直径,所以d_f是self.D减去粗加工的剩余
        """
        Ds, apx2 = [self.D], 0
        for ap in cuts[:-1]:
            apx2 += ap*2
            Ds.append(self.D - apx2)
        Dr, d_f = Ds[:-1], Ds[-1]
        # debug begin
        #~ print('cuts', cuts)
        #~ print('Ds', Ds)
        # debug end
        # debug info
        info = {}
        # debug end
        
        def t(D, L, Vc, f):return 60* 3.14*D*L/(1000.*Vc*f)
        
        t_air_r = [t(d_r, Lair, self.rVc, self.rF) for d_r in Dr]
        t_air_f = t(d_f, Lair, self.fVc, self.fF)
        t_cut_r = [t(d_r, Lcut, self.rVc, self.rF) for d_r in Dr]
        t_cut_f = t(d_f, Lcut, self.fVc, self.fF)
        # debug begin
        info.update(t_cut_r=t_cut_r, t_cut_f=t_cut_f)
        # debug end
        # debug begin
        #~ print('t_air_r')
        #~ print_list(t_air_r)
        #~ print('t_air_f %.3f'%t_air_f)
        #~ print('t_cut_r')
        #~ print_list(t_cut_r)
        #~ print('t_cut_f %.3f'%t_cut_f)
        # debug end
        
        def Est():
            return Pst*(t_st + sum(t_air_r) + t_air_f + sum(t_cut_r) + t_cut_f)
        
        def Eu():
            def pu(Vc,D):return -39.45*(1e3*Vc/(PI*D)) + 0.03125*(1e3*Vc/(PI*D))**2 + 17183
            Pu_r = np.array( [pu(self.rVc, d_r) for d_r in Dr] )
            Pu_f = pu(self.fVc, d_f)
            return Pu_r.dot(t_air_r) + Pu_f*t_air_f + Pu_r.dot(t_cut_r) + Pu_f*t_cut_f
        
        def Emr():
            Pmr = self.Pcut(cuts)
            # debug begin
            info.update(Pmr=Pmr)
            # debug end
            return Pmr[:-1].dot(t_cut_r) + Pmr[-1]*t_cut_f
        
        def Eauc(Pcf=80, Phe=1000):
            return (Pcf+Phe)*(sum(t_air_r) + t_air_f + sum(t_cut_r) + t_cut_f)
        
        #~ def T(Vc, f, Ap):return 60* 4.43*10**12/(Vc**6.8*f**1.37*Ap**0.24)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.33*Ap**0.28)
        def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**0.95*Ap**0.33)
        
        Tr = [T(self.rVc, self.rF, ap_r) for ap_r in cuts[:-1]]
        Tf = T(self.fVc, self.fF, cuts[-1])
        t_ect = np.array(t_cut_r).dot(1/np.array(Tr)) + t_cut_f/Tf
        # debug begin
        #~ print('\nTr', len(Tr))
        #~ for tr_ in Tr:
            #~ print('\t%.3f'%tr_)
        #~ print('Tf %.3f\n'%Tf)
        # debug end
        
        def Ect():
            #~ return Pst*t_pct*t_ect
            return (Pst*t_pct+5340185.)*t_ect
        
        # debug begin
        e_st = Est()
        e_u = Eu()
        e_mr = Emr()
        e_auc = Eauc()
        e_ct = Ect()
        #~ print('t_cut_r vs Dr')
        #~ for t_r, d_r in zip(t_cut_r, Dr):
            #~ print('\t%.3f: %.3f'%(t_r, d_r))
        #~ print('f\t%.3f: %.3f\n'%(t_cut_f, d_f))
        #~ print('e_st: %.3f'% e_st)
        #~ print('e_u: %.3f'% e_u)
        #~ print('e_mr: %.3f'% e_mr)
        #~ print('e_auc: %.3f'% e_auc)
        #~ print('e_ct: %.3f'% e_ct)
        #~ print('Esum: %.3f'% (e_st+e_u+e_mr+e_auc+e_ct) )
        SEC = (e_st+e_u+e_mr+e_auc+e_ct)/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        # debug end
        #~ SEC = (Est()+Eu()+Emr()+Eauc()+Ect())/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        Tp = t_st + sum(t_air_r) + sum(t_cut_r) + t_air_f + t_cut_f + t_pct*t_ect
        
        def Cp(k0=0.3, ke=0.13, Ch=82.5, ne=2, Ci=2.5, N=4):
            kt = Ch/400 + Ci/(0.75*ne)
            #~ return k0*Tp + ke*Tp + kt*t_ect
            return k0*Tp/60 + ke*(e_st+e_u+e_mr+e_auc+e_ct)/3.6e6 + kt*t_ect/N
        
        return SEC, Tp, Cp(N=3), self.penaltyRa(), info
    
    def penaltyRa(self, k=0, u=1, a=2, Ramax=1.6, Kmax=10):
        g = self.rF**2/(8*0.8) - Ramax
        p = a**min(k, Kmax) *u* max(0, g)**2
        return p
    
    def penalty_3(self, a=1e6, t=2, k1=1, k2=2.5, k3=1):
        """ fVc > k1 * rVc   k1 = 1.
        rF  > k2 * fF    k2 = 2.5
        rAp > k3 * fAp   k3 = 1.
        """
        g_fVc = k1 * self.rVc - self.fVc
        g_rF = k2 * self.fF - self.rF
        g_rAp = k3 * self.fAp - self.rAp
        p1 = max(0, g_fVc)**t
        p2 = max(0, g_rF)**t
        p3 = max(0, g_rAp)**t
        return a*(p1+p2+p3)
    
    def reward(self, ):
        """ Normalized reward """
        cuts = self.planning_cuts()
        f0, f1, f2, p0, info = self.f_objectives(cuts)
        objs = np.array([f0, f1, f2])
        #~ objs += p0
        # debug begin
        #~ print('objs', objs)
        #~ print('objs_prev', self.obj_prev)
        # debug end
        if self.obj_prev is None:
            self.obj_prev = objs
        # debug begin
        #~ print('objs', objs)
        #~ print('objs_prev', self.obj_prev)
        # debug end
        Rs = np.sign(self.obj_prev - objs)
        self.obj_prev = objs
        #~ R = -sum(np.tanh(objs)*0.3)
        #~ R = Rs[0]
        # debug begin
        R = np.tanh(-f0)
#         R = -f0
        # debug end
        return R, f0, cuts
    
    def _action_scale(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        act_k = (self.Limit6[1] - self.Limit6[0])/ 2.
        act_b = (self.Limit6[1] + self.Limit6[0])/ 2.
        return act_k * action + act_b
    
    def on_action(self, action, type=0):
        """ States 
        """
        #~ d_fVc, d_fF, d_fAp, d_rVc, d_rF, d_rAp = np.clip(action, *self.Limit6)
        
        #~ self.fVc += d_fVc
        #~ self.fF += d_fF
        #~ self.fAp += d_fAp
        
        #~ self.rVc += d_rVc
        #~ self.rF += d_rF
        #~ self.rAp += d_rAp
        
        #~ self.fVc, self.fF, self.fAp, \
        #~ self.rVc, self.rF, self.rAp = self.make_obs().clip(*self.Limit6)
        
        #~ action = self._action_scale(action)
        
        if type==0:
            self.fVc, self.fF, self.fAp, \
            self.rVc, self.rF, self.rAp = (self.make_obs()+action).clip(*self.Limit6)
        else:
            self.fVc, self.fF, self.fAp, \
            self.rVc, self.rF, self.rAp = self._action_scale(action).clip(*self.Limit6)
        
        # debug begin
        #~ if self.debug:
            #~ print(self.make_obs())
        # debug end
    
    def reset(self, ):
        self.init_states()
        self.reward()
        self.frame_count = 0
        # Generator Begin
#         return self.make_obs(randn=1)
        # Generator End
        return self.make_obs(obj=self.obs_type)
    
    def function(self, action, mo=0, eval=0):
        """ For SwarmPackagePy """
        #~ print(type(action), action)
        self.on_action(action, type=1)
        cuts = self.planning_cuts()
        f0, f1, f2, p0, info = self.f_objectives(cuts)
        mobj = np.array([f0, f1, f2])
        mobj += self.penalty_3()
        obj = f0
        if eval:
            return self.make_obs(), np.array(cuts), info
        if mo:
            return mobj
        return obj
    
    def step(self, action):
        
        info, done = {}, 0
        
        self.on_action(action, type=0)
        #self.on_action(action, type=1)
        
        reward, f0, cuts = self.reward()
        
        #~ if self.check_states(obs) is False:
            #~ done = 1
            #~ reward = -1
            #~ print("Fatal Error: should not happen")
        
        obs = self.make_obs(obj=self.obs_type)
        # Generator Begin
        #obs = self.make_obs(randn=1)
        # Generator End
        
        # limit steps begin
        self.frame_count += 1
        if self.frame_count>=1e3:
            self.frame_count = 0
            done = 1
        # limit steps end
        
        # debug begin
        def get_solution():
            return 'Nc:%d f(Vc%.3f F%.3f Ap%.3f) r(Vc%.3f F%.3f Ap%.3f) obj:%.3f'%\
                    (len(cuts), self.fVc, self.fF, self.fAp, self.rVc, self.rF, self.rAp, f0)
        if self.solution_info is None or f0<self.solution_info[1]:
            solution = get_solution()
            self.solution_info = [solution, f0, 1]
            if self.best_reward is None:
                self.best_reward = f0+1
        #~ if self.debug:
            #~ print('\ncuts %rmm %r reward %.3f'%(sum(cuts), len(cuts), reward))
            #~ print(self.solution_info)
        info.update(f0=f0)
        if done:
            if self.debug:
                print('done', get_solution())
                print('best', self.solution_info)
            if self.best_reward>self.solution_info[1]:
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
    


class MooEnv2(MooEnv):
    def __init__(self, *args, **kwds):
        super(MooEnv2, self).__init__(*args, **kwds)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,) )
        
        #~ self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(3,) )
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,) )
        # best of lifetime
        self.best_solution = dict(obj=None, sol=[])
        self.ref_point = [25, 25000]
        
        self.ep_objs = np.array([0.,0.,0.])
        self.ep_objs_prev = self.ep_objs.copy()
        self.ep_count = 0
        
        self.pop = None
        
        self.single_obj_type = 0
        self.moo = 0
    
    def init_states(self, ):
        self.Dt = self.D
        self.Kt = self.LimitApf[1]
        self.Kf = 0
        self.Stage = 0
        self.tVc = None
        self.tF = None
        self.tAp = None
        
        self.ep_solution = dict(obj=None, sol=[])
        
        self.ep_objs_prev = self.ep_objs.copy()
        self.ep_objs = np.array([0.,0.,0.])
        
        self.step_objs = np.array([0.,0.,0.])
    
    def make_obs(self):
        #~ return np.array( [self.Stage, self.Dt, self.Kt] )
        #~ return np.array( [*self.ep_objs_prev[:2], self.Stage, self.Dt, self.Kt] )
        return np.array( [*self.step_objs[:2], self.Stage, self.Dt, self.Kt] )
    
    def Pcut(self):
        """ x_fix: al,40cr,45,80,55,steel
        """
        x_fix = [0,1,0,0,1,0]
        X = MinMaxScale([self.tVc, self.tF, self.tAp], *self.VFAPminmax)
        Xs = np.append(X, x_fix)
        Xs = np.array([Xs])
        Pc = self.estimator.predict(Xs).clip(*self.LimitPcut)
        # debug begin
        #~ print(Xs.shape, Xs)
        #~ print('Pc', Pc.shape)
        #~ for p in Pc:
            #~ print('\t%.3f'%p)
        # debug end
        return Pc[0]
    
    def f_objectives(self, Lair=15, Lcut=100, Pst=945, t_st=50, t_pct=300):
        """ 注意D是每一刀开始的直径.
        Stage(0)精加工, Stage(1+)粗加工.
        """
        t_st = t_st if self.Stage==1 else 0
        
        Dt = self.Dt
        
        def t(D, L, Vc, f):return 60* 3.14*D*L/(1000.*Vc*f)
        
        t_air = t(Dt, Lair, self.tVc, self.tF)
        t_cut = t(Dt, Lcut, self.tVc, self.tF)
        
        def Est():
            return Pst*(t_st + t_air + t_cut)
        
        def Eu():
            def pu(Vc,D):return -39.45*(1e3*Vc/(PI*D)) + 0.03125*(1e3*Vc/(PI*D))**2 + 17183
            Pu_t = pu(self.tVc, Dt)
            return Pu_t*t_air + Pu_t*t_cut
        
        def Emr():
            Pmr = self.Pcut()
            return Pmr*t_cut
        
        def Eauc(Pcf=80, Phe=1000):
            return (Pcf+Phe)*(t_air + t_cut)
        
        #~ def T(Vc, f, Ap):return 60* 4.43*10**12/(Vc**6.8*f**1.37*Ap**0.24)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.33*Ap**0.28)
        #~ def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**1.03*Ap**0.31)
        def T(Vc, f, Ap):return 60* 4.33*10**12/(Vc**6.9*f**0.95*Ap**0.33)
        
        t_ect = t_cut/T(self.tVc, self.tF, self.tAp)
        
        def Ect():
            """ 只有精加工之后发生 """
            #~ return Pst*t_pct*t_ect *(1 if self.Stage==0 else 0)
            #~ return (Pst*t_pct+5340185.)*t_ect *(1 if self.Stage==0 else 0)
            return (Pst*t_pct+5340185.)*t_ect
        
        # debug begin
        e_st = Est()
        e_u = Eu()
        e_mr = Emr()
        e_auc = Eauc()
        e_ct = Ect()
        
        SEC = (e_st+e_u+e_mr+e_auc+e_ct)/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        # debug end
        #~ SEC = (Est()+Eu()+Emr()+Eauc()+Ect())/(0.785*(self.D**2-self.Dfinal**2)*Lcut)
        Tp = t_st + t_air + t_cut + t_pct*t_ect
        
        def Cp(k0=0.3, ke=0.13, Ch=82.5, ne=2, Ci=2.5, N=4):
            kt = Ch/400 + Ci/(0.75*ne)
            #~ return k0*Tp + ke*Tp + kt*t_ect
            return k0*Tp/60 + ke*(e_st+e_u+e_mr+e_auc+e_ct)/3.6e6 + kt*t_ect/N
        
        return SEC, Tp, Cp(N=3), self.penaltyRa()
    
    def penaltyRa(self, k=0, u=1, a=1e6, Ramax=1.6, Kmax=10):
        if self.Stage==0:
            return 0
        g = self.tF**2/(8*0.8) - Ramax
        p = a**min(k, Kmax) *u* max(0, g)**2
        return p
    
    def penaltyApr(self, ap_r, a=1e6, k=2):
        """ default a=10 """
        gmin = self.LimitApr[0] - ap_r    # min < ap_r
        gmax = ap_r - self.LimitApr[1]    # ap_r < max
        pmin = max(0, gmin)**k
        pmax = max(0, gmax)**k
        return a*(pmin + pmax)
    
    def penalty_3(self, a=1e6, t=2, k1=1, k2=2.5, k3=1):
        """ fVc > k1 * rVc   k1 = 1.
        rF  > k2 * fF    k2 = 2.5
        rAp > k3 * fAp   k3 = 1.
        """
        if self.Stage==0:
            return 0
        g_fVc = k1 * self.tVc - self.fVc
        g_rF = k2 * self.fF - self.tF
        g_rAp = k3 * self.fAp - self.tAp
        p1 = max(0, g_fVc)**t
        p2 = max(0, g_rF)**t
        p3 = max(0, g_rAp)**t
        return a*(p1+p2+p3)
    
    def reward(self, mo=0):
        f0, f1, f2, p0 = self.f_objectives()
        p_apr = self.penaltyApr(self.tAp) if self.Stage>0 else 0
        mobj = np.array([f0, f1, f2])
        # store solution begin
        self.ep_objs += mobj
        # store solution end
        if mo:
            # multiobjective R
            step_obj_now, step_obj_prev = mobj[:2], self.step_objs[:2]
            self.step_objs = mobj.copy()
            mobj += p_apr + self.penalty_3()
            R = -p_apr
        else:
            # single-objective
            mobj += p_apr + self.penalty_3()
            if self.single_obj_type==0:
                R = -mobj[0]
            else:
                R = -math.log(mobj[self.single_obj_type])
        return R, mobj
    
    def reward_hv(self, reward=0):
        if self.pop is None:
            self.pop = np.array( [self.ep_objs[:2]] )
        else:
            test_p = np.vstack( (self.pop, self.ep_objs[:2]) )
            hv = pg.hypervolume(test_p)
            reward = hv.exclusive(len(self.pop), hv.refpoint(offset=0.1))
            if reward>0:
                indices = pg.select_best_N_mo(test_p, 100)
                self.pop = test_p[indices]
                reward = math.tanh(reward)
        return reward
    
    def _action_scale3(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        Limiter = self.Limit3f if self.Stage==0 else self.Limit3r
        act_k = (Limiter[1] - Limiter[0])/ 2.
        act_b = (Limiter[1] + Limiter[0])/ 2.
        return act_k * action + act_b
    
    def begin_step(self, action, eps=0.05):
        self.tVc, self.tF, self.tAp = self._action_scale3(action)
        def get_states():
            return [self.tVc, self.tF, self.tAp, self.Dt, self.Kt+self.Kf, self.Kt+self.Kf-self.tAp]
        assert self.Kt>0
        # Generate State Params Begin
        is_final_roughing = 0
        if self.Stage==0:
            self.Dt = self.Dfinal + self.tAp*2
            self.Kt = self.tAp
            # store solutions begin
            self.ep_solution['sol'].append( get_states() )
            self.Kf = self.Kt
            # store solutions end
            self.fVc, self.fF, self.fAp = self.tVc, self.tF, self.tAp
        else:
            if self.tAp>=self.Kt or (self.Kt-self.tAp)/self.tAp<eps:
                self.tAp = self.Kt
                is_final_roughing = 1
            # store solutions begin
            self.ep_solution['sol'].insert(-1, get_states())
            # store solutions end
        # Generate State Params End
        return is_final_roughing
    
    def end_step(self):
        if self.Stage==0:
            self.Dt = self.D
            self.Kt = self.K - self.tAp
        else:
            self.Dt -= self.tAp*2
            self.Kt -= self.tAp
        self.Stage += 1
    
    def reset(self):
        self.init_states()
        return self.make_obs()
    
    def step(self, action):
        """ action: [Vc,f,Ap] """
        info, done = {}, 0
        
        done = self.begin_step(action)
        reward, _ = self.reward(mo=self.moo)
        self.end_step()
        
        obs = self.make_obs()
        
        if self.moo:
            info.update(pref=self.ref_point)
        # store solutions begin
        if done:
            best_obj = self.best_solution.get('obj', None)
            if best_obj is None or best_obj[self.single_obj_type]>self.ep_objs[self.single_obj_type]:
                self.best_solution.update(obj=self.ep_objs.copy())
                info.update(solution=[self.get_solution(self.ep_objs), self.ep_objs[self.single_obj_type], 1])
            info.update(f0=self.ep_objs[self.single_obj_type])
            info.update(mobj=self.ep_objs.copy()[:2])
            info.update(sol='n_cuts: %d'%len(self.ep_solution['sol']))
            
            if self.debug:
                print('done', self.get_solution(self.ep_objs))
                print('best', self.best_solution)
            
            # moo begin
            if self.moo:
                ep_objs = self.ep_objs[:2]
                ep_objs_prev = self.ep_objs_prev[:2]
                #~ if pg.pareto_dominance(ep_objs, ep_objs_prev):
                    #~ # reward = 1
                    #~ reward = self.reward_hv(reward)
                #~ elif pg.pareto_dominance(ep_objs_prev, ep_objs):
                    #~ reward = -1
                #~ else:
                    #~ # reward = 1
                    #~ reward = self.reward_hv(reward)
                reward = self.reward_hv()
                obs = self.reset()
                self.ep_count += 1
                if self.ep_count%1000==0:
                    done = 1
                else:
                    done = 0
                
                #~ reward = self.reward_hv(reward)
            # moo end
        # store solutions end
        
        return obs, float(reward), done, info
    
    def penaltySumAp(self, sum_ap, n_err, a=1e7, k=2):
        h = abs(sum_ap-self.K)
        #~ return h**k
        #~ return h**k + 2*abs(n_err)**k
        #~ return h**k + 50*abs(n_err)**2
        return a*h**k + a*abs(n_err)**k
    
    def function(self, X, mo=0, penaltly=1, eval=0):
        """ For NiaPy """
        self.init_states()
        
        X = np.array(X).reshape(-1, 3)
        mobj_ep = np.zeros(3)
        n_x = X.shape[0]
        n_sol = 0
        for x in X:
            done = self.begin_step(x)
            _, mobj = self.reward(mo=mo)
            self.end_step()
            mobj_ep += mobj
            n_sol += 1
            if done:
                break
        
        if penaltly:
            sum_ap = self.get_solution(self.ep_objs, getAp=1)
            penaltly_aps = self.penaltySumAp(sum_ap, n_err=n_sol-n_x)
            mobj_ep += penaltly_aps
        
        if eval:
            return self.get_solution(self.ep_objs), None
        
        if mo:
            return mobj_ep
        return mobj_ep[self.single_obj_type]
    
    def get_solution(self, score, getAp=0):
        sol = '----- SEC %.3f, Tp %.3f, Cp %.3f -----\n'%(score[0], score[1], score[2])
        last = len(self.ep_solution['sol'])
        cut_depth = 0
        for i, cut in enumerate(self.ep_solution['sol'], 1):
            cut_depth += cut[2]
            i = 'F' if i==last else str(i)
            #~ sol += '[%s] Vc:%.3f f:%.3f Ap:%.3f\n'%(i, *cut)
            sol += '[%s] Vc:%.3f f:%.3f Ap:%.3f Dt:%.3f Kt:%.3f Kt1:%.3f\n'%((i,)+tuple(cut))
        sol += '------------------ %.3fmm ------------------\n'%cut_depth
        if getAp:
            return cut_depth
        return sol


class MooEnvX(MooEnv):
    def __init__(self, *args, **kwds):
        super(MooEnvX, self).__init__(*args, **kwds)
        
        DIM = 3
        #~ DIM = 2
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(DIM,) )
        
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(DIM,) )
        
        self.obj_prev = None
        self.solution_info = None
        self.best_reward = None
        
        #~ self.sol = np.random.uniform(-1, 1, size=self.observation_space.shape[0])
        self.LimitSol = -500.0, 500.
        self.sol = np.random.uniform(*self.LimitSol, size=self.observation_space.shape[0])
        
        self.pop = None
        
        np.set_printoptions(precision=3)
        
        # debug begin
        self.frame_count = 0
        # debug end
        self._gym_vars()
    
    def make_obs(self, randn=0, reset=0):
        if randn==1:
            return np.random.randn(self.observation_space.shape[0])
        elif randn==2:
            return self.sol + np.random.randn(self.observation_space.shape[0])*1e-5
        elif reset:
            self.sol = np.random.uniform(*self.LimitSol, size=self.observation_space.shape[0])
        return self.sol
    
    def _sol_scale(self, action, Lower, Upper):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        act_k = (Upper - Lower)/ 2.
        act_b = (Upper + Lower)/ 2.
        return act_k * action + act_b
    
    def on_action(self, action, type=0):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if type==0:
            self.sol = self.sol + action
            return self.sol
        self.sol = action
        return action
    
    def reset(self):
        self.frame_count = 0
        #~ return self.make_obs(reset=1)
        return self.make_obs(randn=1)
    
    def step(self, action):
        info, done = {}, 0
        
        action = self.on_action(action, type=1)
        f0, sol = self.SchwefelX(action, type=1)
        
        objs = np.array([f0])
        if self.obj_prev is None:
            self.obj_prev = objs
        Rs = np.sign(self.obj_prev - objs)
        self.obj_prev = objs
        #~ reward = Rs[0]
        reward = -f0
        
        #~ obs = self.make_obs()
        obs = self.make_obs(randn=2)
        
        # limit steps begin
        self.frame_count += 1
        if self.frame_count>=1e3:
            self.frame_count = 0
            done = 1
        #~ done = 1
        # limit steps end
        
        # debug begin
        def get_solution():
            return '%.3f %s'%(f0, str(sol))
        if self.solution_info is None or f0<self.solution_info[1]:
            solution = get_solution()
            self.solution_info = [solution, f0, 1]
            if self.best_reward is None:
                self.best_reward = f0+1
        #~ if self.debug:
            #~ print('\ncuts %rmm %r reward %.3f'%(sum(cuts), len(cuts), reward))
            #~ print(self.solution_info)
        info.update(f0=f0)
        if done:
            if self.debug:
                print('done', get_solution())
                print('best', self.solution_info)
            if self.best_reward>self.solution_info[1]:
                info.update(solution=self.solution_info)
                self.best_reward = self.solution_info[1]
        # debug end
        
        return obs, float(reward), done, info
    
    def SchwefelX(self, sol, Lower=-500.0, Upper=500.0, type=0):
        if type==0:
            sol =self.sol= sol.clip(Lower, Upper)
        else:
            sol = self._sol_scale(sol, Lower, Upper)
        
        val = 0.0
        
        for sol_i in sol:
            val += (sol_i * math.sin(math.sqrt(abs(sol_i))))
        
        return 418.9829 * len(sol) - val, sol




