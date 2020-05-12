import pickle

loss = pickle.load(open('ilocus-v0/driverSim-v0-run66/loss-1of5.p','rb'))

from matplotlib import pyplot

loss_x = [tmp_[0] for tmp_ in loss]
loss_y = [tmp_[1] for tmp_ in loss]
pyplot.plot(loss_x, loss_y)
pyplot.show()