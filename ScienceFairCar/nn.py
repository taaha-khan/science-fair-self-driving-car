# Imports
import random, math, pickle

# Network Sums (Numpy Library)
class Matrix:

	# Constructor Initializing
	def __init__(self, rows, cols):
		self.rows = rows
		self.cols = cols
		self.data = []

		# Initializing Matrix
		for i in range(self.rows):
			self.data.append([])
			for _ in range(self.cols):
				self.data[i].append(0)

	# Copying Matrix
	def copy(self):
		m = Matrix(self.rows, self.cols)
		for i in range(self.rows):
			for j in range(self.cols):
				m.data[i][j] = self.data[i][j]
		return m

	# Creating Matrix from Array
	def fromArray(self, arr):
		m = Matrix(len(arr), 1)
		for i in range(len(arr)):
			m.data[i][0] = arr[i]
		return m

	# Creating Array from Matrix
	def toArray(self):
		arr = []
		for i in range(self.rows):
			for j in range(self.cols):
				arr.append(self.data[i][j])
		return arr

	# Randomizing Matrix
	def randomize(self):
		for i in range(self.rows):
			for j in range(self.cols):
				self.data[i][j] = (random.random() * 2) - 1

	# Adding to Matrix
	def add(self, n):
		if isinstance(n, Matrix):
			for i in range(self.rows):
				for j in range(self.cols):
					self.data[i][j] += n.data[i][j]
		else:
			for i in range(self.rows):
				for j in range(self.cols):
					self.data[i][j] += n
	
	# Getting New Matrix (a - b)
	def static_subtract(self, a, b):
		result = Matrix(a.rows, a.cols)
		for i in range(result.rows):
			for j in range(result.cols):
				result.data[i][j] = a.data[i][j] - b.data[i][j]
		return result

	# Reversing Matrix Dimensions
	def static_transpose(self, matrix):
		result = Matrix(matrix.cols, matrix.rows)
		for i in range(matrix.rows):
			for j in range(matrix.cols):
				result.data[j][i] = matrix.data[i][j]
		return result

	# Multiplying Value to Matrix
	def multiply(self, n):
		if isinstance(n, Matrix):  # Hadamard Product
			for i in range(self.rows):
				for j in range(self.cols):
					self.data[i][j] *= n.data[i][j]
		else:  # Scalar Product
			for i in range(self.rows):
				for j in range(self.cols):
					self.data[i][j] *= n

	# Apply function to all elts of Matrix
	def map(self, function):
		for i in range(self.rows):
			for j in range(self.cols):
				val = self.data[i][j]
				self.data[i][j] = function(val)

	# Applying function to copy of Matrix
	def static_map(self, matrix, function):
		result = Matrix(matrix.rows, matrix.cols)
		for i in range(matrix.rows):
			for j in range(matrix.cols):
				val = matrix.data[i][j]
				result.data[i][j] = function(val)
		return result
	
	# Raising Matrix Values to Power
	def exp(self, num):
		for i in range(self.rows):
			for j in range(self.cols):
				val  = self.data[i][j]
				self.data[i][j] = val ** num

	# Getting Sum of all Values in Matrix
	def sum(self):
		total = 0
		for i in range(self.rows):
			for j in range(self.cols):
				total += self.data[i][j]
		return total

	# Multiply Matrices
	def static_multiply(self, a, b):
		if a.cols != b.rows:
			print('\n\nArray Invalid\n\n')
			raise ValueError
		else:  # Multiplying other Matrix
			result = Matrix(a.rows, b.cols)
			for i in range(result.rows):
				for j in range(result.cols):
					total = 0
					for k in range(a.cols):  
						total += a.data[i][k] * b.data[k][j]
					result.data[i][j] = total
			return result
	
	def mutate(self, rate):
		for i in range(self.rows):
			for j in range(self.cols):
				if random.random() < rate:
					self.data[i][j] = (2 * random.random()) - 1
	
	def println(self):
		data = ''
		for j in range(self.cols):
			for i in range(self.rows):
				val = self.data[i][j]
				data += str(val)
				data += '  '
			data += '\n'
		print(data)

# Activation Function
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# Derivative of Sigmoided Value
def dsigmoid(y):
	# return sigmoid(x) * (1 - sigmoid(x))
	return y * (1 - y)

# Network
class NeuralNetwork:

	# Constructor
	def __init__(self, a, b, c):
		
		# Copy Same Neural Network
		if isinstance(a, NeuralNetwork):
			self.input_nodes = a.input_nodes
			self.hidden_nodes = a.input_nodes
			self.output_nodes = a.input_nodes

			self.weights_ih = a.weights_ih.copy()
			self.weights_ho = a.weights_ho.copy()

			self.bias_h = a.bias_h.copy()
			self.bias_o = a.bias_o.copy()


		else:  # New Neural Network

			# Network Node Amounts
			self.input_nodes = a
			self.hidden_nodes = b
			self.output_nodes = c

			# Weight Matrices
			self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
			self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
			self.weights_ih.randomize()
			self.weights_ho.randomize()

			# Bias Matrices
			self.bias_h = Matrix(self.hidden_nodes, 1)
			self.bias_o = Matrix(self.output_nodes, 1)
			self.bias_h.randomize()
			self.bias_o.randomize()
			self.learning_rate = 0.1


	# Feedforward Algorithm
	def predict(self, input_array):

		# Generating Hidden Outputs
		inputs = Matrix.fromArray(Matrix, input_array)
		hidden = Matrix.static_multiply(Matrix, self.weights_ih, inputs)
		hidden.add(self.bias_h)

		# Activation Function
		hidden.map(sigmoid)
		
		# Generating Output
		output = Matrix.static_multiply(Matrix, self.weights_ho, hidden)
		output.add(self.bias_o)
		output.map(sigmoid)

		return output.toArray()

	# Normalizing Outputs to 100%
	def softmax(self, outputs, A):
		outputs = outputs.exp(A)
		return outputs / outputs.sum()
	
	# Backpropagation Training Algorithm
	def train(self, input_array, target_array):

		# FEEDFORWARD ALGORITHM ---------

		# Generating Hidden Outputs
		inputs = Matrix.fromArray(Matrix, input_array)
		hidden = Matrix.static_multiply(Matrix, self.weights_ih, inputs)
		hidden.add(self.bias_h)

		# Activation Function
		hidden.map(sigmoid)
		
		# Generating Output
		outputs = Matrix.static_multiply(Matrix, self.weights_ho, hidden)
		outputs.add(self.bias_o)
		outputs.map(sigmoid)

		# -------------------------------

		# Converting Arrays to Matrix Objects
		targets = Matrix.fromArray(Matrix, target_array)

		# Calculating Errors
		output_errors = Matrix.static_subtract(Matrix, targets, outputs)
		
		# Calculating Gradient
		gradients = Matrix.static_map(Matrix, outputs, dsigmoid)
		gradients.multiply(output_errors)
		gradients.multiply(self.learning_rate)

		# Calculate Deltas
		hidden_T = Matrix.static_transpose(Matrix, hidden)
		weigths_ho_deltas = Matrix.static_multiply(Matrix, gradients, hidden_T)

		# Adjusting Weights & Bias (hidden -> output)
		self.weights_ho.add(weigths_ho_deltas)
		self.bias_o.add(gradients)

		# Calculating Hidden Layer Errors
		who_t = Matrix.static_transpose(Matrix, self.weights_ho)
		hidden_errors = Matrix.static_multiply(Matrix, who_t, output_errors)
		
		# Calculate Hidden Gradient
		hidden_gradient = Matrix.static_map(Matrix, hidden, dsigmoid)
		hidden_gradient.multiply(hidden_errors)
		hidden_gradient.multiply(self.learning_rate)

		# Calculate (Input -> Hidden) Deltas
		inputs_T = Matrix.static_transpose(Matrix, inputs)
		weight_ih_deltas = Matrix.static_multiply(Matrix, hidden_gradient, inputs_T)

		# Adjusting Weights (input -> hidden)
		self.weights_ih.add(weight_ih_deltas)
		self.bias_h.add(hidden_gradient)

	def dump_weights(self, filepath):
		with open(filepath, 'wb') as file:
			pickle.dump({
				'ih': self.weights_ih.data,
				'ho': self.weights_ho.data,
				'bh': self.bias_h.data,
				'bo': self.bias_o.data
			}, file)
	
	def load_weights(self, filepath):
		with open(filepath, 'rb') as file:
			weights = pickle.load(file) 
		self.weights_ih.data = weights['ih']
		self.weights_ho.data = weights['ho']
		self.bias_h.data = weights['bh']
		self.bias_o.data = weights['bo']

	# Neuro-evolution Functions

	# Copying same Neural Network
	def copy(self):
		return NeuralNetwork(self, None, None)
	
	# Mutating Child Neural Network Weights
	def mutate(self, rate):

		self.weights_ih.mutate(rate)
		self.weights_ho.mutate(rate)
		self.bias_h.mutate(rate)
		self.bias_o.mutate(rate)
