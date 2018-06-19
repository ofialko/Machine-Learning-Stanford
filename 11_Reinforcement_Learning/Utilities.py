import gym
from gym import wrappers
import numpy as np

class RandomAgent(object):
    def __init__(self,env,n_policy):
        self.env = env
        self.n_policy = n_policy
        self.obs_space = env.observation_space
        self.act_space = env.action_space

        if  type(self.obs_space) == gym.spaces.Discrete and type(self.act_space) == gym.spaces.Discrete:
            self.type = "DD"
            self.x = self.obs_space.n
            self.y = self.act_space.n
            self.obs = np.zeros(self.x)
        elif type(self.obs_space) == gym.spaces.Discrete and type(self.act_space) == gym.spaces.Box:
            self.type = "DB"
            self.x = self.obs_space.n
            self.y = self.act_space.shape[0]
            self.obs = np.zeros(self.x)
        elif type(self.obs_space) == gym.spaces.Box and type(self.act_space) == gym.spaces.Discrete:
            self.type = "BD"
            self.x = self.obs_space.shape[0]
            self.y = self.act_space.n
            self.obs = np.zeros(self.x)

        elif type(self.obs_space) == gym.spaces.Box and type(self.act_space) == gym.spaces.Box:
            self.type = "BB"
            self.x = self.obs_space.shape[0]
            self.y = self.act_space.shape[0]
            self.obs = np.zeros(self.x)
        else:
            raise Exception('Observation/Action space is not supported')

        self.best_policy = None

    def softmax(self,vals):
        x = np.exp(vals)
        return x/np.sum(x)

    def gen_W(self):
        '''
        Creates mapping matrix from obs to action
        '''
        return np.random.randn(self.x,self.y)

    def gen_obs(self,obs):
        if self.type == 'DD' or self.type == 'DB':
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
        if self.type == 'DD' or self.type == 'BD':
            return np.argmax(np.matmul(obs,W))
        else:
            return np.matmul(obs,W)


    def run_episode(self,policy=None, outdir = None, t_max=2000):
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
