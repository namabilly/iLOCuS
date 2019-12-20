import numpy as np
import random
import os
import pickle
import math
import copy

class Driver:    
    def __init__(self, x, y):
        self.grid_x = x
        self.grid_y = y
        self.itinerary = []

# the main class for the simulator
# here's an example of interacting with it
# drivers = Drivers() # constructor
# drivers.reset(8, 1000, "20151101")  # start at 8am in 11/01/2015, simulate with 1000 taxis
# drivers.step(np.ones((15,15))) # call this at each timestep with a bonus matrix
class Drivers:
    LNG_MIN = 116.3002190000
    LNG_MAX = 116.4802271600
    LNG_GRID = 15
    LAT_MIN = 39.8493175200
    LAT_MAX = 39.9836824300
    LAT_GRID = 15
    LNG_STEP = (LNG_MAX - LNG_MIN) / LNG_GRID
    LAT_STEP = (LAT_MAX - LAT_MIN) / LAT_GRID
    MOVE = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self):
        self.time_idx = 0
        self.drivers = []
        self.empties = []
        self.requests = {}
        self.empty_10min = []
        self.request_10min = []

    def reset(self, time = 8, count = 1000, date = "20151101"):
        random.seed(None)
        self.time_idx = time * 30
        self.drivers = []

        dir_path = os.path.join(os.path.dirname(__file__), "../requests_2min")
        req_path = os.path.join(dir_path, date + "_request_list.pickle")
        with open(req_path, 'rb') as f:
            self.requests = pickle.load(f)

        # Deprecated: use request distribution as taxi distribution
        # sample_requests = [[0 for j in range(self.LNG_GRID)] for i in range(self.LAT_GRID)]
        # total_requests = 0
        # for i in range(self.LAT_GRID):
        #     for j in range(self.LNG_GRID):
        #         for k in range(8 * 30, 12 * 30):
        #             sample_requests[i][j] += len(self.requests[k][i][j])
        #             total_requests += len(self.requests[k][i][j])

        dir_path = os.path.join(os.path.dirname(__file__), "../matrices_10min/")
        req_path = os.path.join(dir_path, date + "_request.npy")
        self.request_10min = np.load(req_path)
        emp_path = os.path.join(dir_path, date + "_free.npy")
        self.empty_10min = np.load(emp_path)
        data = self.empty_10min[self.time_idx // 5]
        total = data.sum()
        distribution = [[float(data[i][j] / total) for j in range(self.LNG_GRID)] for i in range(self.LAT_GRID)]

        taxi_count = [[round(count * distribution[i][j]) for j in range(self.LNG_GRID)] for i in range(self.LAT_GRID)]
        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                for k in range(taxi_count[i][j]):
                    self.drivers.append(Driver(i, j))
        return self.state()

    def inside(self, x, y):
        return x >= 0 and x < self.LAT_GRID and y >= 0 and y < self.LNG_GRID

    def load_requests(self):
        # randomly generated requests for testing
        generated = [[[] for j in range(self.LNG_GRID)] for i in range(self.LAT_GRID)]
        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                for k in range(random.randrange(5)): # number of requests in a grid
                    start = (random.randrange(self.LAT_GRID), random.randrange(self.LNG_GRID))
                    request = [start]
                    for l in range(random.randrange(10)): # length of trip
                        lst = request[-1]
                        mov = self.MOVE[random.randrange(4)]
                        nxt = (lst[0] + mov[0], lst[1] + mov[1])
                        if self.inside(nxt[0], nxt[1]):
                            request.append(nxt)
                        else:
                            request.append(lst)
                    generated[i][j].append(request)
        return generated

    def state(self):
        taxi_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        empty_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        request_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        occupied_count = 0
        for driver in self.drivers:                   
            taxi_count[driver.grid_x][driver.grid_y] += 1
            if not driver.itinerary: 
                empty_count[driver.grid_x][driver.grid_y] += 1
            else:
                occupied_count += 1

        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                request_count[i][j] = len(self.requests[self.time_idx][i][j])
                
        print("Time idx: %d, Occupied Rate: %.2f" % (self.time_idx, occupied_count / len(self.drivers)))
        ret = np.zeros((3, self.LAT_GRID, self.LNG_GRID))
        ret[0,:,:] = request_count
        ret[1,:,:] = taxi_count
        ret[2,:,:] = empty_count
        return ret, False

    def get_utility(self, i, j):
        req_cnt = self.request_10min[self.time_idx // 5][i][j]
        emp_cnt = self.empty_10min[self.time_idx // 5][i][j]
        
        # avg = time / cnt # average request length
        probility = req_cnt / emp_cnt if req_cnt < emp_cnt else 1 # probability of getting a request
        # TODO: choose the right value for this constant
        constant = 10 # a constant to control the relative value compared to bonus, or to say an expected earn per unit

        return probility * constant
    
    def step(self, bonus):
        # advance time
        self.time_idx += 1
        # count empty taxis in grids to assign requests
        self.empties = [[[] for i in range(self.LAT_GRID)] for j in range(self.LNG_GRID)]
        for driver in self.drivers:            
            if not driver.itinerary: 
                # if not occupied: add for lottery
                self.empties[driver.grid_x][driver.grid_y].append(driver)

        # lottery pick to match requests & drivers
        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                requests = list(self.requests[self.time_idx][i][j].values())
                req_cnt = self.request_10min[self.time_idx // 5][i][j]
                emp_cnt = self.empty_10min[self.time_idx // 5][i][j]                
                if len(requests) == 0 or req_cnt == 0: 
                    continue
                # print(req_cnt, emp_cnt, len(requests), len(self.empties[i][j]))
                for driver in self.empties[i][j]:
                    # roll the dice, probability = available requests / empty taxis, from the original dataset
                    if emp_cnt == 0 or random.randrange(emp_cnt) < req_cnt:
                        driver.itinerary = copy.deepcopy(random.choice(requests))
                # disable this for now
                # for t in range(max(0, self.time_idx - 5), self.time_idx):
                #     # requests could be left for 5 timesteps
                #     requests = self.requests[t]
                #     if len(requests[i][j]) > 0 and len(taxis[i][j]) > 0:
                #         for key, request in requests[i][j].items():
                #             if len(taxis[i][j]) == 0:
                #                 break
                #             random.shuffle(taxis[i][j])
                #             driver = taxis[i][j].pop()
                #             driver.itinerary = request

        # make a move for all drivers
        for driver in self.drivers:            
            if not driver.itinerary: 
                # still not occupied: go to the best possible adjacent grids, myopic driver
                cx, cy = driver.grid_x, driver.grid_y
                candidates = [(cx, cy)]
                creward = bonus[cx][cy] + self.get_utility(cx, cy)
                # TODO: all unoccupied taxis goes in the same direction?
                for i in range(4):
                    nx, ny = cx + self.MOVE[i][0], cy + self.MOVE[i][1]
                    if self.inside(nx, ny):
                        reward = bonus[nx][ny] + self.get_utility(nx, ny)
                        if math.isclose(reward, creward):
                            candidates.append((nx, ny))
                        elif reward > creward:
                            candidates = [(nx, ny)]
                            creward = reward
                        
                dest = random.choice(candidates)
                driver.grid_x, driver.grid_y = dest
            else:
                # currently occupied: go to the next grid in itinerary
                # for starting rides, driver will stay in the current grid (time cost of take request)
                # for finishing rides, driver will become available in next step (time cost of dropping)
                next_grid = driver.itinerary.pop(0)
                driver.grid_x, driver.grid_y = next_grid[0], next_grid[1]

        # compute new state for output
        return self.state()
                        
   

# test its functionality
# data = np.load('./matrices_10min/20151101_free.npy')
# print(len(data))
# print(data[100])
# print(data[100].sum())
drivers = Drivers()
drivers.reset(8, 1000)
for i in range(400):
    drivers.step(np.ones((15,15)))
# drivers.step(np.zeros((15,15)))

