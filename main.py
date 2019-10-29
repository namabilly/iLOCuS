
import numpy as np
import datetime
import time
from model import *
from view import *

data = np.load('20151101.npy')

model = Model()
# view = View(model)

# time utc 1101 - 1446336000
# 8 hours diff - 1446307200

currtime = 1446307200
# store info where vehicles pick up passengers
filteredarray = []
while currtime < 1446393600:
	# currdata = filter(lambda x: currtime <= x[1] < currtime + 60, data)
	currdata = data[np.logical_and(currtime <= data[:, 1], data[:, 1] < currtime + 120)]
	# print(currdata)
	
	# matrix = [[0 for x in range(10)] for y in range(10)] 
	statearray = []
	
	for d in currdata:
		model.add_agent(d[0], d[1], (d[2], d[3]), d[6], None, statearray)
	
	# print(currtime, statearray)
	# load filtered data
	for array in statearray:
		filteredarray.append(array)
	# view.draw()
	currtime += 120
	# time.sleep(0.01)

filteredarray = np.array(filteredarray, dtype='float')
np.save('20151101_filtered.npy', filteredarray)

