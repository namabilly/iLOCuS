


class Model:

	p_accept = 0.8 # prob of agents accepting incentive
	# can move it to agent for specific prob
	
	# size of the map, might need some casting for position from GPS
	# num of agents, 
	def __init__(self):
		self.agents = {}
		self.num_of_agents = 0
		self.map = []
		
	def add_agent(self, id, position, status, sensor):
		self.agents[id] = Agent(position, status, sensor)
		self.num_of_agents += 1
	
	def get_distribution(self):
		''' pseudo
		distribution = []
		for pos in self.map:
			distribution[pos]
			count the sensors
		return distribution
		'''
		
	# should take traffic into account, posbly use gps api
	def get_trajectories(self, source, target):
		return []
		
	# assume json
	def update(self, data):
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
		
		
	
	