
# Model - the simulation
# not to be confused with prediction

class Model:
	
	# size of the map, might need some casting for position from GPS
	# num of agents, 
	def __init__(self):
		self.agents = {}
		self.num_of_agents = 0
		self.map = []
		
	def add_agent(self, id, position, status, sensor):
		self.agents[id] = Agent(position, status, sensor)
		self.num_of_agents += 1
	
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
		
	def get_distribution(self):
		''' pseudo
		distribution = []
		for pos in self.map:
			distribution[pos]
			count the sensors
		return distribution
		or return map
		'''
		
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
		
	
	
	