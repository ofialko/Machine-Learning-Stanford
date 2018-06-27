
from Agents import *
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')

    outdir = 'Results'
    #agent = RandomAgent(env,1000)
    agent = EvolutionAgent(env,20,50)

    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
