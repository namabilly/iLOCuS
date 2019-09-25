
import numpy as np
import datetime
import time
from model import *
from view import *

data = np.load('20151101.npy')

model = Model()
view = View(model)

# time utc 1101 - 1446336000
# 8 hours diff - 1446307200

currtime = 1446307200
while currtime < 1446393600:
	# currdata = filter(lambda x: currtime <= x[1] < currtime + 60, data)
	currdata = data[np.logical_and(currtime <= data[:, 1], data[:, 1] < currtime + 60)]
	# print(currdata)
	for d in currdata:
		model.add_agent(d[0], (d[2], d[3]), d[6], None)
	# print(model.agents)
	view.draw()
	time.sleep(1)




