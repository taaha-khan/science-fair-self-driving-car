
''' Q-Network Library '''

from collections import defaultdict
from graph import Graph
import numpy as np
import os
import nn

class QNetwork:

	def __init__(self):

		self.epsilon = 1
		self.min_epsilon = 0.05
		self.epsilon_decay = 0.94

		self.actions = list(range(3))
		self.model = nn.NeuralNetwork(30, 200, len(self.actions))

		self.history = []
		self.num = 0

	def step(self, state):

		state = self.normalize(state)

		predictions = self.model.predict(state)
		action = int(np.argmax(predictions))

		# print(np.array(predictions))

		return action
	
	def train(self, data):

		old_q = self.model.predict(data['state'])
		new_q = self.model.predict(data['next_state'])

		old_q[data['action']] = (data['reward'] + max(new_q)) / 2
		if data['reward'] == 0:
			old_q[data['action']] = 0

		X = data['state'].copy()
		Y = old_q.copy()

		self.model.train(X, Y)

	def update(self, env, raw_state):

		# Generating action
		if np.random.random_sample() < self.epsilon:
			action = np.random.choice(self.actions)
		else:
			action = self.step(raw_state)
		
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
		for _ in range(5):
			self.train(np.random.choice(self.history))
		
		# Reducing history to only last 100 steps
		if len(self.history) > 100:
			self.history.pop(0)

		return raw_next_state, reward, done, info
	
	def learn(self, env, save_to, loaded = 'none', num_episodes = 200):

		if os.path.exists(loaded):
			self.load(loaded)
		
		graph = Graph('Reward vs Episodes', 'Episode', 'Reward')
		avg_reward = 0
		
		for episode in range(1, num_episodes + 1):

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
			
			avg_reward += total_reward
			graph.step(episode, avg_reward / episode)

			self.save(save_to)

			print(f'episode {episode} completed | steps {self.num} | reward: {total_reward} | avg: {avg_reward / episode}')
		print(f'\n{num_episodes} training episodes completed')

		graph.show()

	def normalize(self, state):
		return state[1].flatten().tolist()

	def save(self, filepath):
		self.model.dump_weights(filepath)
		# print(f'model saved to {filepath}')

	def load(self, filepath):
		self.model.load_weights(filepath)
		print(f'model loaded from {filepath}')
		return self.model