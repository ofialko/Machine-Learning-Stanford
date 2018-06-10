'''
Classical problem in RL: Solving the CartPole environment.
It simulates an inverted pendulum mounted upwards on cart,
the pendulum is intially vertical and the goal is to maintain it vertically balanced.
The only way to control the pendulum is by choosing a horizontal direction for the cart to move
either to left or right. Two solutions are provided.
'''

import numpy as np
import gym
from gym import wrappers

# ## 1.0 Introducing CartPole
env = gym.make('CartPole-v0')
# Restart the environment to start a new episode
env.reset()

for step_idx in range(500):
    env.render()
    obs, reward, done, _ = env.step(env.action_space.sample())
    #if done:
    #    env.reset()
env.close()

# ## 1.1 First attempt to solve the CartPole problem
# Identify vectors with bad policy

X = []
env = gym.make('CartPole-v0')

for step_idx in range(10000):
    obs = env.reset()
    while True:
        obs, reward, done, _ = env.step(env.action_space.sample())
        if done:
            X.append(obs.tolist())
            break

X = np.array(X)


# Make a step.
# If the minimal distance between new obserbvation and bad policy vectors decreases
# change the step
env = gym.make('CartPole-v0')
obs = env.reset()
#start with vertical position for simpplicity
env.env.state = np.zeros(4)
d0=np.linalg.norm(X - obs,axis=1).max()
s=env.action_space.sample()

for step_idx in range(200):
    env.render()
    obs, reward, done, _ = env.step(s)
    d = np.linalg.norm(X-obs,axis=1).min()
    if d <= d0:
        s = (s+1)%2
    d0 = d
    if done:
        pass
        #break

env.close()


# ## 1.2 Random search

def gen_random_policy():
    return (np.random.uniform(-1,1, size=4), np.random.uniform(-1,1))

def policy_to_action(policy, obs):
    if np.dot(policy[0], obs) + policy[1] > 0:
        return 1
    else:
        return 0

def run_episode(env, policy, t_max=1000, render=False):
    obs = env.reset()
    total_reward = 0
    for i in range(t_max):
        if render:
            env.render()
        selected_action = policy_to_action(policy, obs)
        obs, reward, done, _ = env.step(selected_action)
        total_reward += reward
        if done:
            break
    return total_reward


env = gym.make('CartPole-v0')

## Generate a pool or random policies
n_policy = 500
policy_list = [gen_random_policy() for _ in range(n_policy)]

# Evaluate the score of each policy.
scores_list = [run_episode(env, p) for p in policy_list]

# Select the best policy.
print('Best policy score = %f' %max(scores_list))

best_policy= policy_list[np.argmax(scores_list)]
print('Running with best policy:\n')
run_episode(env, best_policy, render=True)

env.close()