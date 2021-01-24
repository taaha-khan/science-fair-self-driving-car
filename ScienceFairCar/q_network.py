
# https://github.com/satwikkansal/q-learning-taxi-v3

import matplotlib.pyplot as plt
from matplotlib.pyplot import show, plot
from collections import defaultdict
import numpy as np
import pickle
import os
import nn

class QNetwork:

	def __init__(self):

		self.epsilon = 1
		self.min_epsilon = 0.05
		self.epsilon_decay = 0.94

		self.actions = list(range(3))
		self.model = nn.NeuralNetwork(90, 200, len(self.actions))

		self.history = []
		self.num = 0

	def step(self, env, state):

		state = self.normalize(state)

		predictions = self.model.predict(state)
		action = int(np.argmax(predictions))

		# print(list(map(lambda a: round(a, 2), predictions)))

		return action
	
	def train(self, data):

		old_q = np.array(self.model.predict(data['state']))
		new_q = self.model.predict(data['next_state'])

		old_q[data['action']] = data['reward'] * max(new_q)

		X = data['state'].copy()
		Y = list(old_q / old_q.max())

		self.model.train(X, Y)

	def update(self, env, raw_state):

		# Generating action
		if np.random.random_sample() < self.epsilon:
			action = np.random.choice(self.actions)
		else:
			action = self.step(env, raw_state)
		
		# Stepping through environment
		raw_next_state, reward, done, info = env.step(action)

		# Normalizing data
		state = self.normalize(raw_state)
		next_state = self.normalize(raw_next_state)
		
		# Adding state to memory
		self.history.append({
			'state': state,
			'next_state': next_state,
			'action': action,
			'reward': reward
		})

		# Training from last step
		self.train(self.history[-1])

		# Training from random batch from previous steps
		self.train(np.random.choice(self.history))

		# Only remembering previous 100 steps
		if len(self.history) > 100:
			self.history.pop(0)

		return raw_next_state, reward, done, info
	
	def learn(self, env, loaded = 'q_network_200', num_episodes = 200):

		if os.path.exists(loaded):
			self.load(loaded)
		
		fig = plt.gcf()
		fig.show()
		fig.canvas.draw()

		y_axis = []
		
		for episode in range(num_episodes):

			state = env.reset()

			total_reward = 0
			for _ in range(env.config['duration']):
				state, reward, done, _ = self.update(env, state)
				env.render()
				total_reward += reward
				self.num += 1
				if done: 
					break
			
			self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

			x_axis = list(range(episode + 1))
			y_axis.append(total_reward)

			plt.plot(x_axis, y_axis, color = 'red', marker = 'o')
			plt.title('Reward vs Episodes')
			plt.xlabel('Episodes')
			plt.ylabel('Reward')

			plt.xlim([0, max(x_axis) + 1])
			plt.ylim([0, max(y_axis) + 1])

			fig.canvas.draw()

			print(f'episode {episode} completed | steps {self.num} | reward: {total_reward}')
		print(f'\n{num_episodes} training episodes completed')

		self.save(loaded)

		plt.show()
	
	def normalize(self, state):
		if type(state) not in [list, tuple]:
			return state.flatten().tolist()
		return state

	def save(self, filepath = f'q_network_200'):
		self.model.dump_weights(filepath)
		print(f'model saved to {filepath}')

	def load(self, filepath = f'q_network_200'):
		self.model.load_weights(filepath)
		print(f'model loaded from {filepath}')
		return self.model