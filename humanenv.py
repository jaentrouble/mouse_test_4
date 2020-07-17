import gym
import gym_mouse

env = gym.make('mouseCl-v0')
env.seed(3)
env.reset()
total_reward = 0
while True :
    env.render()
    a = int(input('Move:'))
    if a == -1 :
        break
    o, r, d, i = env.step(a)
    total_reward += 1
    print('reward : {}'.format(r))
    if d :
        env.reset()
        print('done, total reward:{}'.format(total_reward))
        total_reward = 0
