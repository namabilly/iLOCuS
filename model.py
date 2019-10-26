import math
import numpy as np
import time
# Model - the simulation
# not to be confused with prediction

class Model:
	
	# size of the map, might need some casting for position from GPS
	# num of agents, 
	def __init__(self):
		self.agents = {}
		self.num_of_agents = 0
		self.map = []
		self.prediction_pos = {}
	
	def add_agent(self, id, time, position, status, sensor, matrix):
		# status change
		id = int(id)
		if id in self.agents:
			if self.agents[id].status == 1 and status == 0:
				if matrix != None:
					# x, y = self.map_coordinate(position, [[39.60, 40.25], [116.04, 116.88]], [len(matrix), len(matrix[0])])
					# matrix[x][y] += 1
					matrix.append([id, time, position[0], position[1], status])					
		self.agents[id] = Agent(position, status, sensor)
		self.num_of_agents += 1
	
	def map_coordinate(self, position, range, slice):
		incre_x = (range[0][1] - range[0][0])/slice[0]
		incre_y = (range[1][1] - range[1][0])/slice[1]
		x = math.floor((position[0] - range[0][0])/incre_x)
		y = math.floor((position[1] - range[1][0])/incre_y)
		return (x, y)
		
	# give random normalized matrix
	def normalize_matrix(self, matrix, sigma):
		np.random.seed(math.floor(time.time()))
		for x in range(len(matrix)):
			for y in range(len(matrix[0])):
				matrix[x][y] = np.random.normal(matrix[x][y], sigma, None)
				# matrix[x][y] = math.ceiling(matrix[x][y])
				if matrix[x][y] < 0:
					matrix[x][y] = 0
		return matrix
	
	# give random normalized route positions
	def normalize_route(self, routes, sigma):
		# sigma recommended to be fairly small due to logitude/latitude
		np.random.seed(math.floor(time.time()))
		for x in range(len(routes)):
			routes[x][0] = np.random.normal(routes[x][0], sigma, None)
			routes[x][1] = np.random.normal(routes[x][1], sigma, None)
		return routes
	
	# return prob of agents accepting incentive
	def p_agent_accept(self, id, target, incentive):
		pos = self.agents[id].position
		# time, path = min(self.get_trajactories(pos, target))
		# return (incentive - threshold) / (time * k)
		
	# simulate how the agents react to the incentive orders
	def react_agents(self, orders):
		# note that the order should be optimized through crowdsourcer,
		# may give same location to multiple agents (?)
		# should one agent receive more than one?
		# for order in orders:
		#	determine what each agent does
		# self.update()
		return True
		
	def get_distribution(self):
		''' pseudo
		distribution = []
		for pos in self.map:
			distribution[pos]
			count the sensors
		return distribution
		or return map
		'''
		return map
		
	# should take traffic into account, posbly use gps api
	def get_trajectories(self, source, target):
		# simulation
		# Maybe we need another class to generate these data
		# get a map of routes and num of cars
		return []
		
	# assume json
	def update(self, data):
		self.map = []
		for ag in data.agents:
			self.agents[ag.id].update(ag.position, ag.status)
			# may cast position
			self.map[ag.position].append(id)


class Agent:

	def __init__(self, position, status, sensor):
		self.position = position
		self.starting_pos = position
		self.status = status
		'''
		0 - free
		1 - occupied
		2 - incentivized
		'''
		self.sensor = sensor
		
	def update(self, position, status):
		self.position = position
		self.status = status
		
	# visualization
	def draw(self, canvas):
		# whole Beijing
		# 39.44 - 41.06
		# 115.42 - 117.50
		
		# in this section as of 20151101
		# 39.60 - 40.25
		# 116.04 - 116.88
		
		x = (self.position[1] - 116.04) * 500 / .84
		y = (self.position[0] - 39.60) * 500 / .65
		color = 'green' if self.status == 0 else 'pink'
		point = canvas.create_oval(x, y, x+1, y+1, outline=color, fill=color)
	
	