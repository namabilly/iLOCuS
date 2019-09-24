
# transform data into numpy array
import numpy as np
import pickle
filename = '20151101'
with open(filename + '.txt', 'r') as file:
	data = []
	line = file.readline()
	count = 0 # 2m line ~ 1GB memory
	while line and count < 10000000:
		data.append(line[:-1].split(','))
		line = file.readline()
		count += 1
'''
dataline = data.split('\n')
mat = list(map(lambda x: x.split(','), dataline))
data = np.array(mat)
'''
data = np.array(data, dtype='float')
np.save(filename + '.npy', data)

# seems that data are too large for the memory to execute above code? 

