import numpy as np
import random
import os

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
    CNT_GRID = LNG_GRID * LAT_GRID
    PRICE_PER_KM = 0.60 # gas or overall?
    CIRCUITY_FACTOR = 1.34 # 1.2 - 1.6
    MOVE = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def __init__(self):
        self.time_idx = 0
        self.drivers = []

    def reset(self, time = 9, count = 2000, date = "20151101"):
        random.seed(None)
        self.time_idx = time * 30
        self.drivers = []
        for i in range(count):
            # TODO: initial position needed
            self.drivers.append(Driver(random.randrange(self.LAT_GRID), random.randrange(self.LNG_GRID)))
        
        # dir_path = os.path.join(os.path.dirname(__file__), "../matrices_10min")
        # req_path = os.path.join(dir_path, date + "_request.npy")
        # self.ride_requests = np.load(req_path)
        return self.state()

    def inside(self, x, y):
        return x >= 0 and x < self.LAT_GRID and y >= 0 and y < self.LNG_GRID

    def load_requests(self, time):
        # TODO: fetch for this timestep
        generated = [[[] for i in range(self.LAT_GRID)] for j in range(self.LNG_GRID)]
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


    def state(self, requests = None):
        taxi_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        empty_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        request_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        for driver in self.drivers:                   
            taxi_count[driver.grid_x][driver.grid_y] += 1
            if requests is not None:
                request_count[driver.grid_x][driver.grid_y] = len(requests[driver.grid_x][driver.grid_y])
            if not driver.itinerary: 
                # not occupied (add for lottery)
                empty_count[driver.grid_x][driver.grid_y] += 1
                
        ret = np.zeros((4, self.LAT_GRID, self.LNG_GRID))
        ret[0,:,:] = request_count
        ret[2,:,:] = taxi_count
        ret[3,:,:] = empty_count
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
        requests = self.load_requests(self.time_idx)
        for i in range(self.LAT_GRID):
            for j in range(self.LNG_GRID):
                if len(requests[i][j]) > 0 and len(taxis[i][j]) > 0:
                    for request in requests[i][j]:
                        if len(taxis[i][j]) == 0:
                            break
                        random.shuffle(taxis[i][j])
                        driver = taxis[i][j].pop()
                        driver.itinerary = request
                        # print(i, j, driver.itinerary)

        # make a move for all drivers
        for driver in self.drivers:            
            if not driver.itinerary: 
                # still not occupied: go to the best possible adjacent grids, myopic driver
                cx, cy = driver.grid_x, driver.grid_y
                for i in range(4):
                    nx, ny = driver.grid_x + self.MOVE[i][0], driver.grid_y + self.MOVE[i][1]
                    if self.inside(nx, ny) and bonus[nx][ny] > bonus[cx][cy]:
                        cx, cy = nx, ny
                driver.grid_x, driver.grid_y = cx, cy
            else:
                # currently occupied
                # for starting rides, driver will stay in the current grid (time cost of take request)
                # for finishing rides, driver will become available in next step (time cost of dropping)
                next_grid = driver.itinerary.pop(0)
                driver.grid_x, driver.grid_y = next_grid[0], next_grid[1]

        # compute new state for output
        return self.state(requests)
                        

    

# test its functionality
# [720 * [15 * 15]]
# data = np.load('../dataset/20151101_request.npy')
# for i in data:
#     print(i)
#     break
# print(len(data))
# print(data[300])
# print(data[360])
drivers = Drivers()
drivers.reset(1, 2000)
#drivers.load_requests(1212)
drivers.step(np.ones((15,15)))
drivers.step(np.zeros((15,15)))