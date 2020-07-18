from sanity_env import EnvTest
import gym
import gym_mouse
import numpy as np
from mptestagent import Player
import agent_assets.A_hparameters as hp
from tqdm import trange
import argparse
import os
import sys
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('-mp', dest='mixed_fp',action='store_true', default=False)
parser.add_argument('--step', dest='step', default=10000)
parser.add_argument('-pf', dest='profile', action='store_true', default=False)
args = parser.parse_args()

if args.mixed_fp:
    policy = mixed_precision.Policy('mixed_float16')
else :
    policy = mixed_precision.Policy('float32')
mixed_precision.set_policy(policy)

hp.Buffer_size = 500
hp.Learn_start = 200
hp.Batch_size = 32
hp.Target_update = 500
hp.epsilon = 1
hp.epsilon_min = 0.01
hp.epsilon_nstep = 500

original_env = gym.make('mouseCl-v0')
test_env = EnvTest(original_env.observation_space)
player = Player(original_env.observation_space, test_env.action_space)
o = test_env.reset()
if args.profile:
    for step in trange(hp.Learn_start+50, ncols=100):
        action = player.act(o, training=True)
        o, r, d, i = test_env.step(action)
        player.step(action,r,d,i)
        if d :
            o = test_env.reset()
    with tf.profiler.experimental.Profile('log/profile'):
        for step in trange(10, ncols=100):
            action = player.act(o, training=True)
            o, r, d, i = test_env.step(action)
            player.step(action,r,d,i)
            if d :
                o = test_env.reset()
else :
    for step in trange(int(args.step), ncols=100):
        action = player.act(o, training=True)
        o, r, d, i = test_env.step(action)
        player.step(action,r,d,i)
        if d :
            o = test_env.reset()
