
''' Main execution script '''

# Environment
import highway_env
import gym
from graph import Graph

# Agent lib
import q_network

# Initializing and configuring env
env = gym.make('highway-v0')
env.configure({

	'observation': {
		'type': 'TimeToCollision',
		'horizon': 10
	},

	'collision_reward': 0,
	'lanes_count': 4,

	'duration': 1000,
	'offscreen_rendering': False,

	'screen_width': 1000,
	'screen_height': 350,
	'scaling': 10

})

# Q-Network Actor
agent = q_network.QNetwork()

# Training an agent
# agent.learn(env, save_to = 'saved_model_3', loaded = 'saved_model_2')

# Loading an already trained agent
agent.load('saved_model_2')

# Setting up an environment
env.seed(60)
obs = env.reset()

# Graphing running reward
# graph = Graph('Reward vs Steps', 'Step', 'Reward')

# Executing an evaluation round
for i in range(env.config['duration']):
	action = agent.step(obs)
	obs, reward, done, _ = env.step(action)
	env.render()
	if done:
		break	

	# graph.step(i, running_reward)

# Keeping the screen active
while True:
	env.render()