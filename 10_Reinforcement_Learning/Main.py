
from Agents import *
if __name__ == '__main__':
    env = gym.make('Acrobot-v1')
    outdir = 'Results'
    #agent = RandomAgent(env,500)
    agent = EvolutionAgent(env,100,10)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
