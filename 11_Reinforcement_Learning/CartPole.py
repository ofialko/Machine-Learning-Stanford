'''
Classical problem in RL: Solving the CartPole environment.
It simulates an inverted pendulum mounted upwards on cart,
the pendulum is intially vertical and the goal is to maintain it vertically balanced.
The only way to control the pendulum is by choosing a horizontal direction for the cart to move
either to left or right. Random Search solution is provided.
'''

import gym
from gym import wrappers
import numpy as np

# env = gym.make('FrozenLake-v0')
# obs = env.reset()
# act=env.action_space.sample()
# x = env.step(act);x
#
# # env.reset()
# env.observation_space.sample()
# env.action_space
#
#
#
# env = gym.make('FrozenLake-v0')
# env.reset()
# env.observation_space
# env.action_space.sample()
#
# env = gym.make('MountainCar-v0')
# env.reset()
# env.observation_space.high
# env.action_space.sample()


class RandomAgent(object):
    def __init__(self,env,n_policy):
        self.env = env
        self.n_policy = n_policy
        self.obs_space = env.observation_space
        if len(self.obs_space.shape) == 0:
            self.type = "Discrete"
            self.obs = np.zeros(self.obs_space.n)
        elif len(self.obs_space.shape) == 1:
            self.type = "Box"
            self.obs = np.zeros(self.obs_space.shape)
        else:
            raise Exception('Observatio space is not supported')

        self.act_space = env.action_space
        self.best_policy = None

    def softmax(self,vals):
        x = np.exp(vals)
        return x/np.sum(x)

    def gen_W(self):
        '''
        Creates mapping matrix from obs to action
        '''
        if len(self.obs_space.shape) == 0:
            # if Discrete
            return np.random.randn(self.obs_space.n,self.act_space.n)
        elif len(self.obs_space.shape) == 1:
            # if Box
            return np.random.randn(self.obs_space.shape[0],self.act_space.n)
        else:
            raise ValueError

    def gen_obs(self,obs):
        if self.type == 'Discrete':
            x = self.obs.copy()
            x[obs] = 1
            return x
        else:
            return obs
    # def gen_random_policy(self):
    #     return (self.observation_space.sample(), np.random.uniform(-1,1))

    def gen_policy_list(self):
        ## Generate a pool or random policies
        return [self.gen_W() for _ in range(self.n_policy)]

    def policy_to_action(self,W, obs):
        return np.argmax(self.softmax(np.matmul(obs,W)))


    def run_episode(self,policy=None, outdir = None, t_max=1000):
        if policy is None:
            policy = self.best_policy
            self.env = wrappers.Monitor(self.env,directory=outdir,force=True)
        obs = self.env.reset()
        total_reward = 0
        for i in range(t_max):

            obs_1 = self.gen_obs(obs)
            selected_action = self.policy_to_action(policy, obs_1)
            obs, reward, done, _ = self.env.step(selected_action)
            total_reward += reward
            if done:
                self.env.close()
                break
        return total_reward

    def train(self):
        policy_list = self.gen_policy_list()
        # Evaluate the score of each policy.
        scores_list = [self.run_episode(policy=p) for p in policy_list]
        # Select the best plicy.
        print('Best policy score = %f' %max(scores_list))
        self.best_policy = policy_list[np.argmax(scores_list)]


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    outdir = '11_Reinforcement_Learning/CartPole-results'
    agent = RandomAgent(env,500)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
