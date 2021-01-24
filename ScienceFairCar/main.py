
import gym
import highway_env

import q_agent
import q_network

env = gym.make("highway-v0")
env.configure({

	'observation': {
		'type': 'TimeToCollision',
		'horizon': 10
	},

	'lanes_count': 4,

	'duration': 25,
	'offscreen_rendering': False,

	'screen_height': 350,
	'screen_width': 1000,
	'scaling': 10,

})

# agent = q_agent.QAgent()
agent = q_network.QNetwork()

agent.learn(env)
agent.load()

obs = env.reset()

for i in range(30):
	action = agent.step(env, obs)
	# action = env.action_space.sample()
	obs, reward, done, info = env.step(action)
	env.render()
	if done:
		break

while True:
	env.render()