import numpy as np
import random
import os
import pickle
import math
import copy
#import seaborn as sns
import matplotlib.pylab as plt
import time

max_turn = 400
save_graph = False
save_data = False
generating = True
req_pcell = 0
driver_pcell = 100

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
    LNG_GRID = 5
    LAT_MIN = 39.8493175200
    LAT_MAX = 39.9836824300
    LAT_GRID = 5
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
        self.counter = 0
        self.dest_count = []
        self.divs = []

    def reset(self, time = 8, count = 1000, date = "20151101"):
        random.seed(None)
        self.time_idx = time * 30
        self.drivers = []

        dir_path = os.path.join(os.path.dirname(__file__), "../requests_2min")
        req_path = os.path.join(dir_path, date + "_request_list.pickle")
        # print(pickle.HIGHEST_PROTOCOL)
        with open(req_path, 'rb') as f:
            self.requests = pickle.load(f)
        
        if generating:
            self.requests[240] = [[{} for j in range(self.LNG_GRID)] for i in range(self.LAT_GRID)]
            for i in range(self.LAT_GRID):
                for j in range(self.LNG_GRID):
                    self.requests[240][i][j] = {}
            pool = [(i, j) for i in range(self.LAT_GRID) for j in range(self.LNG_GRID)]
            pool = [val for val in pool for _ in range(req_pcell)]
            for i in range(self.LAT_GRID):
                for j in range(self.LNG_GRID):
                    for k in range(req_pcell):
                        random.shuffle(pool)
                        ex, ey = pool.pop()
                        # ex, ey = (random.randrange(15), random.randrange(15))
                        path = self.path_to((i, j), (ex, ey))
                        ind = 0
                        while ind in self.requests[240][i][j]:
                            ind += 1
                        self.requests[240][i][j][ind] = path
        
        self.dest_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        start_count = np.zeros((self.LAT_GRID, self.LNG_GRID))
        for requests in self.requests[self.time_idx]:
            for req in requests:
                for key in req:
                    x, y = req[key][-1]
                    self.dest_count[x][y] += 1
                    x, y = req[key][0]
                    start_count[x][y] += 1
        # print(self.dest_count)
        
        if save_data:
            np.save(os.path.join('result', '20151101_' + str(self.time_idx) + '_request_start.npy'), start_count)
            np.save(os.path.join('result', '20151101_' + str(self.time_idx) + '_request_dest.npy'), self.dest_count)
        
        if save_graph:
            ax = sns.heatmap(self.dest_count, linewidth=0.5)
            plt.savefig(os.path.join('plots', '20151101_' + str(self.time_idx) + '_request_dest'))
            plt.clf()
            ax = sns.heatmap(start_count, linewidth=0.5)
            plt.savefig(os.path.join('plots', '20151101_' + str(self.time_idx) + '_' + str(self.counter) + '_request_start'))
            plt.clf()
            
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
        for i in range(1):
            for j in range(1):
                for k in range(driver_pcell if generating else taxi_count[i][j]): 
                    self.drivers.append(Driver(i, j))
        return self.state()

    def inside(self, x, y):
        return x >= 0 and x < self.LAT_GRID and y >= 0 and y < self.LNG_GRID

    # Deprecated
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

    def path_to(self, a, b):
        dx = 1 if a[0] < b[0] else -1
        dy = 1 if a[1] < b[1] else -1
        path = []
        cx, cy = a[0], a[1]
        path.append((cx, cy))
        while cx != b[0] or cy != b[1]:
            if random.random() < 0.5:
                if cx != b[0]:
                    cx += dx
                else:
                    cy += dy
            else:
                if cy != b[1]:
                    cy += dy
                else:
                    cx += dx
            path.append((cx, cy))
        return path

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
                
        # print("Time idx: %d %d, Occupied Rate: %.2f" % (self.time_idx, self.counter, occupied_count / len(self.drivers)))
        # print(taxi_count)
        
        if self.counter % 10 == 0 and self.counter < 210:
            if save_data:
                np.save(os.path.join('result', '20151101_' + str(self.time_idx) + '_' + str(self.counter) + '_request_taxi.npy'), taxi_count)        
            if save_graph:
                ax = sns.heatmap(taxi_count, linewidth=0.5)
                plt.show()
                plt.savefig(os.path.join('plots', '20151101_' + str(self.time_idx) + '_' + str(self.counter) + '_request_taxi'))
                plt.clf()
        
        ret = np.zeros((3, self.LAT_GRID, self.LNG_GRID))
        ret[0,:,:] = request_count
        ret[1,:,:] = taxi_count
        ret[2,:,:] = empty_count
        
        # divergence = self.KL(taxi_count, self.dest_count)
        # print("divergence is: %.10f" % (divergence))
        # self.divs.append(divergence)
        
        return ret

    def get_utility(self, i, j):
        # replace with actual req/emp currently, will be historical values afterwards
        #req_cnt = self.request_10min[self.time_idx // 5][i][j]
        #emp_cnt = self.empty_10min[self.time_idx // 5][i][j]
        req_cnt = len(self.requests[self.time_idx][i][j])
        emp_cnt = 0
        for driver in self.drivers:                   
            if driver.grid_x == i and driver.grid_y == j and not driver.itinerary: 
                emp_cnt += 1
        # avg = time / cnt # average request length
        probility = req_cnt / (emp_cnt + 1) # if req_cnt < emp_cnt else 1 # probability of getting a request
        # TODO: choose the right value for this constant
        constant = 10 # a constant to control the relative value compared to bonus, or to say an expected earn per unit

        return probility * constant
    
    def softmax(self, arr, bonus):
        # choose among candidates with softmax
        l = len(arr)
        val = [(bonus[a[0]][a[1]] + self.get_utility(a[0], a[1])) for a in arr]
        sum = 0
        prob = []
        for v in val:
            sum += pow(math.e, v)
            prob.append(sum)
        prob /= sum
        n = np.random.random()
        for i in range(l):
            if n <= prob[i]:
                return arr[i]
        return arr[0]

    def step(self, bonus):
        print("starting each step...")
        for i in range(20):
            self.each_step(bonus)
        print("ending step state...")
        print(self.state()[1,:,:])

        return self.state()


    def each_step(self, bonus):
        # advance time
        #self.time_idx
        self.counter += 1
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
                for driver in self.empties[i][j]:
                    # roll the dice, probability = available requests / empty taxis, from the original dataset
                    if np.random.random() * len(self.empties[i][j]) < len(requests):
                        rn = math.floor(np.random.random() * len(requests))
                        driver.itinerary = copy.deepcopy(requests[rn])
                        
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
                        candidates.append((nx, ny))
                    '''
                        reward = bonus[nx][ny] + self.get_utility(nx, ny)
                        
                        if math.isclose(reward, creward):
                            candidates.append((nx, ny))
                        elif reward > creward:
                            candidates = [(nx, ny)]
                            creward = reward
                        
                        if math.isclose(reward, creward) or reward > creward:
                            candidates.append((nx, ny))
                    '''
                #dest = candidates[math.floor(np.random.random() * len(candidates))]
                dest = self.softmax(candidates, bonus)
                driver.grid_x, driver.grid_y = dest
            else:
                # currently occupied: go to the next grid in itinerary
                # for starting rides, driver will stay in the current grid (time cost of take request)
                # for finishing rides, driver will become available in next step (time cost of dropping)
                next_grid = driver.itinerary.pop(0)
                driver.grid_x, driver.grid_y = next_grid[0], next_grid[1]

        # compute new state for output
        return self.state()
                        
    def KL(self, state, objective):
        state = np.copy(state) + 1e-7
        state /= np.sum(state)
        objective = np.copy(objective) + 1e-7
        objective /= np.sum(objective)
        # KL divergence
        return -np.sum(np.where(state != 0, state * np.log(state / objective), 0))

# test its functionality
# data = np.load('./matrices_10min/20151101_free.npy')
# print(len(data))
# print(data[100])
# print(data[100].sum())
drivers = Drivers()
drivers.reset(8, 1000)
for i in range(max_turn):
    drivers.step(np.ones((5,5)))
# drivers.step(np.zeros((15,15)))
y = drivers.divs
if save_graph:
    plt.plot(y)
    plt.savefig(os.path.join('plots', 'generative_divergence'))
    plt.clf()
