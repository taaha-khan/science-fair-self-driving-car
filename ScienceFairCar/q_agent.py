
''' Q-Table Library '''

from collections import defaultdict
import numpy as np
import random
import pickle
import os

class QAgent:

	def __init__(self):

		self.alpha = 0.1
		self.gamma = 0.6

		self.epsilon = 1
		self.min_epsilon = 0.05
		self.epsilon_decay = 0.94

		self.episode_length = 30

		self.q_table = defaultdict(lambda: {i: 0 for i in range(5)})

		self.actions = {
			'LANE_LEFT': 0,
			'IDLE': 1,
			'LANE_RIGHT': 2,
			'FASTER': 3,
			'SLOWER': 4
		}
	
	def step(self, env, state):

		state = self.normalize(state)

		max_q_value_action = env.action_space.sample()
		max_q_value = 0

		if state in self.q_table:
			print(f'Checked: {list(self.q_table[state].values())}')
			for action, action_q_value in self.q_table[state].items():
				if action_q_value > max_q_value:
					max_q_value = action_q_value
					max_q_value_action = action
		else:
			print('Random Action')

		return max_q_value_action

	def update(self, env, raw_state):

		if random.random() < self.epsilon:
			action = env.action_space.sample()
			print('Random Action')
		else:
			action = self.step(env, raw_state)
		
		next_state, reward, done, info = env.step(action)

		next_state = self.normalize(next_state)
		state = self.normalize(raw_state)

		old_q_value = self.q_table[state][action]

		# Check if next_state has q values already
		if next_state not in self.q_table:
			self.q_table[next_state] = {
				action: 0 for action in range(env.action_space.n)
			}

		# Maximum q_value for the actions in next state
		next_max = max(self.q_table[next_state].values())

		# Calculate the new q_value
		new_q_value = (1 - self.alpha) * old_q_value + self.alpha * (reward + self.gamma * next_max)

		# Finally, update the q_value
		self.q_table[state][action] = new_q_value

		return next_state, reward, done, info
	
	def learn(self, env, loaded = 'q_table', num_episodes = 500):

		if os.path.exists(loaded):
			self.load(loaded)

		self.num_episodes = num_episodes

		for episode in range(num_episodes):

			state = env.reset()
			normal = self.normalize(state)
			if normal not in self.q_table:
				self.q_table[normal] = {
					action: 0 for action in range(env.action_space.n)
				}

			total_reward = 0
			for _ in range(env.config['duration']):
				state, reward, done, _ = self.update(env, state)
				env.render()
				total_reward += reward
				if done: 
					break
			
			self.epsilon *= self.epsilon_decay
			self.epsilon = max(self.epsilon, self.min_epsilon)
			
			print(f'episode {episode} completed | reward: {total_reward} | epsilon {self.epsilon}')
		print(f'\n{num_episodes} training episodes completed')

		self.save()
		return self.q_table
	
	def normalize(self, state):
		return hash(str(state))

	def save(self, filepath = f'q_table'):
		with open(filepath, 'wb') as file:
			pickle.dump(dict(self.q_table), file)
		# print(f'model saved to {filepath}')
		
	def load(self, filepath = f'q_table'):
		with open(filepath, 'rb') as file:
			self.q_table = pickle.load(file) 
		print(f'model loaded from {filepath}')
		return self.q_table