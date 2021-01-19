
import gym
import highway_env

import q_agent

env = gym.make("highway-v0")
env.configure({
	'observation': {
		'type': 'TimeToCollision',
		'horizon': 5
	},

	'duration': 25,
	'simulation_frequency': 15,
	'offscreen_rendering': False,

	'screen_height': 350,
	'screen_width': 1000,
	'scaling': 10,

})

agent = q_agent.QAgent()
agent.learn(env)
# agent.load()

# env.seed(q_agent.SEED)
obs = env.reset()

for i in range(30):
	action = agent.step(env, obs)
	# action = env.action_space.sample()
	obs, reward, done, info = env.step(action)
	# print(obs)
	env.render()
	if done:
		break

while True:
	env.render()