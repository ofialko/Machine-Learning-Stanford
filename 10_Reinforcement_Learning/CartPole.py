'''
Classical problem in RL: Solving the CartPole environment.
It simulates an inverted pendulum mounted upwards on cart,
the pendulum is intially vertical and the goal is to maintain it vertically balanced.
The only way to control the pendulum is by choosing a horizontal direction for the cart to move
either to left or right. Random Search solution is provided.
'''

from Utilities import RandomAgent, gym
if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    outdir = '11_Reinforcement_Learning/CartPole-results'
    agent = RandomAgent(env,500)
    agent.train()
    print('Training Done')
    reward = agent.run_episode(outdir = outdir)
    env.close()
