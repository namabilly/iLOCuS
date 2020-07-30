import pickle

loss = pickle.load(open('driverSim-v0-run157/loss-3of5.p','rb'))

from matplotlib import pyplot
loss_x = [tmp_[0] for tmp_ in loss]
loss_y = [tmp_[1] for tmp_ in loss]
pyplot.plot(loss_x, loss_y)
pyplot.show()