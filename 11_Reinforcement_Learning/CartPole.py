'''
Classical problem in RL: Solving the CartPole environment.
It simulates an inverted pendulum mounted upwards on cart,
the pendulum is intially vertical and the goal is to maintain it vertically balanced.
The only way to control the pendulum is by choosing a horizontal direction for the cart to move
either to left or right. Two solutions are provided.
'''

import gym
from gym import wrappers
import numpy as np

class RandomAgent(object):
    def __init__(self,env):
        self.env = env
        self.observation_space = env.observation_space
        self.best_policy = self.observation_space.sample()

    def gen_random_policy(self):
        return (self.observation_space.sample(), np.random.uniform(-1,1))

    def gen_policy_list(self,n_policy):
        ## Generate a pool or random policies
        return [self.gen_random_policy() for _ in range(n_policy)]

    def policy_to_action(self,policy, obs):
        if np.dot(policy[0], obs) + policy[1] > 0:
        	return 1
        else:
        	return 0

    def run_episode(self,policy=None, outdir = None, t_max=1000):
        if policy is None:
            policy = self.best_policy
            self.env = wrappers.Monitor(self.env,directory=outdir,force=True)
        obs = self.env.reset()
        total_reward = 0
        for i in range(t_max):
            selected_action = self.policy_to_action(policy, obs)
            obs, reward, done, _ = self.env.step(selected_action)
            total_reward += reward
            if done:
                self.env.close()
                break
        return total_reward

    def train(self):
        n_policy = 500
        policy_list = self.gen_policy_list(n_policy)
        # Evaluate the score of each policy.
        scores_list = [self.run_episode(policy=p) for p in policy_list]
        # Select the best plicy.
        print('Best policy score = %f' %max(scores_list))
        self.best_policy = policy_list[np.argmax(scores_list)]


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    outdir = '11_Reinforcement_Learning/CartPole-results'
    agent = RandomAgent(env)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
