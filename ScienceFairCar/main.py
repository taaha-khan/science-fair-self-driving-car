
''' Main execution script '''

# Environment
import highway_env
import gym

# Agent lib
import q_network

# Initializing and configuring env
env = gym.make("highway-v0")
env.configure({

	'observation': {
		'type': 'TimeToCollision',
		'horizon': 10
	},

	'collision_reward': 0,
	'lanes_count': 4,

	'duration': 1000,
	'offscreen_rendering': False,

	'screen_height': 350,
	'screen_width': 1000,
	'scaling': 10,

})

# Q-Network Actor
agent = q_network.QNetwork()

# Training the agent
# agent.learn(env)

# Loading a trained agent
agent.load('saved_model')

# Setting up an environment
env.seed(60)
obs = env.reset()

# Executing an evaluation round
for i in range(env.config['duration']):
	action = agent.step(obs)
	obs, _, done, _ = env.step(action)
	env.render()
	if done:
		break

# Keeping the screen active
while True:
	env.render()