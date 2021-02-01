
import matplotlib.pyplot as plt

class Graph:
	
	def __init__(self, title = '', xlabel = '', ylabel = ''):

		self.x_axis = []
		self.y_axis = []

		self.title = title
		self.xlabel = xlabel
		self.ylabel = ylabel

		self.fig = plt.gcf()
		self.fig.show()
		self.fig.canvas.draw()
	
	def step(self, x, y):

		self.x_axis.append(x)
		self.y_axis.append(y)
		
		plt.plot(self.x_axis, self.y_axis, color = 'red')

		plt.title(self.title)
		plt.xlabel(self.xlabel)
		plt.ylabel(self.ylabel)

		plt.xlim([min(self.x_axis) - 1, max(self.x_axis) + 1])
		plt.ylim([min(self.y_axis) - 1, max(self.y_axis) + 1])

		self.fig.canvas.draw()
	
	def show(self):
		plt.show()