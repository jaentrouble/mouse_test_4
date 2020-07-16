from sanity_env import EnvTest
import gym
import gym_mouse
import numpy as np
from Agent import Player
import agent_assets.A_hparameters as hp
from tqdm import trange
import argparse
import os
import sys

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
# for step in trange(1000) :
#     player.act(o,training=True)
#     if step%5 == 0 :
#         action = 2
#     elif step%5 == 1 :
#         action = 1
#     elif step%5 == 2 :
#         action = 2
#     elif step%5 == 3 :
#         action = 1
#     elif step%5 == 4 :
#         action = 0
#     o, r, d, i = test_env.step(action)
#     player.step(action, r,d,i)
#     if d :
#         o = test_env.reset()
for step in trange(10000, ncols=100):
    action = player.act(o, training=True)
    o, r, d, i = test_env.step(action)
    player.step(action,r,d,i)
    if d :
        o = test_env.reset()
    # if step % 1000 == 0 :
    #     print('Evaluating')
    #     vo = test_env.reset()
    #     rewards = 0
    #     for _ in trange(50):
    #         vaction = player.act(vo, training=False)
    #         print(vaction)
    #         vo, vr, vd, vi = test_env.step(vaction)
    #         print(vr)
    #         rewards += vr
    #         if vd :
    #             vo = test_env.reset()
    #     print(rewards/10)
    #     input('continue?')