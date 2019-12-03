import numpy as np
import random
import os
import pickle
import math

class Driver:    
    def __init__(self, x, y):
        self.grid_x = x
        self.grid_y = y
        self.itinerary = []

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
        self.requests = {}

    def reset(self, time = 8, count = 1000, date = "20151101"):
        random.seed(None)
        self.time_idx = time * 30
        self.drivers = []

        dir_path = os.path.join(os.path.dirname(__file__), "../requests_2min")
        req_path = os.path.join(dir_path, date + "_request_list.pickle")
        with open(req_path, 'rb') as f:
            self.requests = pickle.load(f)

        # TODO: initial position needed
        # for i in range(count):
        #     self.drivers.append(Driver(random.randrange(self.LAT_GRID), random.randrange(self.LNG_GRID)))
        # Now: use request distribution as taxi distribution
        sample_requests = [[0 for j in range(self.LNG_GRID)] for i in range(self.LAT_GRID)]
        total_requests = 0

        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                for k in range(8 * 30, 12 * 30):
                    sample_requests[i][j] += len(self.requests[k][i][j])
                    total_requests += len(self.requests[k][i][j])

        distribution = [[sample_requests[i][j] / total_requests for j in range(self.LNG_GRID)] for i in range(self.LAT_GRID)]
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
        for driver in self.drivers:                   
            taxi_count[driver.grid_x][driver.grid_y] += 1
            if not driver.itinerary: 
                empty_count[driver.grid_x][driver.grid_y] += 1

        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                request_count[i][j] = len(self.requests[self.time_idx][i][j])
                
        ret = np.zeros((3, self.LAT_GRID, self.LNG_GRID))
        ret[0,:,:] = request_count
        ret[1,:,:] = taxi_count
        ret[2,:,:] = empty_count
        return ret, False

    
    def step(self, bonus):
        # advance time
        self.time_idx += 1
        # count taxis in grids to assign requests
        taxis = [[[] for i in range(self.LAT_GRID)] for j in range(self.LNG_GRID)]
        # search for empty taxis
        for driver in self.drivers:            
            if not driver.itinerary: 
                # if not occupied: add for lottery
                taxis[driver.grid_x][driver.grid_y].append(driver)

        # lottery pick to match requests & drivers
        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                for t in range(max(0, self.time_idx - 5), self.time_idx): 
                    # requests could be left for 5 timesteps
                    requests = self.requests[t]
                    if len(requests[i][j]) > 0 and len(taxis[i][j]) > 0:
                        for key, request in requests[i][j].items():
                            if len(taxis[i][j]) == 0:
                                break
                            random.shuffle(taxis[i][j])
                            driver = taxis[i][j].pop()
                            driver.itinerary = request

        # make a move for all drivers
        for driver in self.drivers:            
            if not driver.itinerary: 
                # still not occupied: go to the best possible adjacent grids, myopic driver
                cx, cy = driver.grid_x, driver.grid_y
                candidates = [(cx, cy)]
                weights = [bonus[cx][cy]]
                # weighted sample so that not all taxis goes in the same direction
                for i in range(4):
                    nx, ny = cx + self.MOVE[i][0], cy + self.MOVE[i][1]
                    if self.inside(nx, ny):
                        weights.append(bonus[nx][ny])
                        candidates.append((nx,ny))
                weights = [math.exp(x) for x in weights]
                total = sum(weights)
                weights = [x / total for x in weights]
                dest = random.choices(population = candidates, weights = weights)[0]
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
# [720 * [15 * 15]]
# data = np.load('../dataset/20151101_request.npy')
# print(len(data))
# print(data[360])
# drivers = Drivers()
# drivers.reset(8, 1000)
# drivers.step(np.ones((15,15)))
# drivers.step(np.zeros((15,15)))

