from http.client import HTTPSConnection
import numpy as np
import os
import sys
import traceback
import json
# grids index on map
#...             224
#...
#15  16  17  ... 29
#0   1   2   ... 14

class Distance:
    LNG_MIN = 116.3002190000
    LNG_MAX = 116.4802271600
    LNG_GRID = 15
    LAT_MIN = 39.8493175200
    LAT_MAX = 39.9836824300
    LAT_GRID = 15
    LNG_STEP = (LNG_MAX - LNG_MIN) / LNG_GRID
    LAT_STEP = (LAT_MAX - LAT_MIN) / LAT_GRID
    CNT_GRID = LNG_GRID * LAT_GRID
    FILE_NAME = 'distance.npy'

    AK = '0HyPaP3OsuZCP62rU6QZzwtxHkGzY87E'
    HOST = 'api.map.baidu.com'
    DRIVE_API = '/direction/v2/driving?origin=%.6f,%.6f&destination=%.6f,%.6f&ak=%s' # lat, long, lat, long, AK
    CONVERT_API = '/geoconv/v1/?coords=%.6f,%.6f&from=1&to=5&ak=%s' # long, lat, AK

    def __init__(self):
        file_path = os.path.join(os.path.dirname(__file__), self.FILE_NAME)
        self.connection = HTTPSConnection(self.HOST, timeout = 3.0)
        if os.path.exists(file_path):
            self.distance = np.load(file_path)
        else:
            self.distance = np.zeros((self.CNT_GRID, self.CNT_GRID)) # 225*225
            self.generate_distance()
            np.save(file_path, self.distance)

    def get_index_from_rowcol(self, x, y):
        return x * self.LAT_GRID + y

    def get_index_from_coordinate(self, lat, lng):
        LAT_STT, LAT_END = self.LAT_MIN - self.LAT_STEP, self.LAT_MIN
        row_idx, column_idx = -1, -1
        for i in range(self.LAT_GRID):
            LAT_STT += self.LAT_STEP
            LAT_END += self.LAT_STEP
            if LAT_STT <= lat and lat < LAT_END:
                row_idx = i
                break

        LNG_STT, LNG_END = self.LNG_MIN - self.LNG_STEP, self.LNG_MIN
        for j in range(self.LNG_GRID):
            LNG_STT += self.LNG_STEP
            LNG_END += self.LNG_STEP
            if LNG_STT <= lng and lng < LNG_END:
                column_idx = i
                break
        if row_idx == -1 or column_idx == -1:
            print("Error")
            return -1

        return self.get_index_from_rowcol(row_idx, column_idx)


    def coord_convert(self, s_lat, s_lng):
        try:
            url = self.CONVERT_API % (s_lng, s_lat, self.AK)
            self.connection.request('GET', url)
            response = self.connection.getresponse()
            if response.status != 200:    
                raise Exception()
            
            resp_str = response.read().decode('utf-8')
            resp_dict = json.loads(resp_str)  
            s_lat, s_lng = resp_dict["result"][0]["y"], resp_dict["result"][0]["x"]
            return (s_lat, s_lng)
        except:
            traceback.print_exc()
            sys.exit()

    def path_find(self, s_lat, s_lng, d_lat, d_lng):
        try:
            url = self.DRIVE_API % (s_lat, s_lng, d_lat, d_lng, self.AK)
            self.connection.request('GET', url)
            response = self.connection.getresponse()
            if response.status != 200:    
                raise Exception()
            resp_str = response.read().decode('utf-8')
            resp_dict = json.loads(resp_str)
            return resp_dict["result"]["routes"][0]["distance"]
        except:
            traceback.print_exc()
            return self.path_find(s_lat, s_lng, d_lat, d_lng)

    def generate_distance(self):
        LNG_STEP = (self.LNG_MAX - self.LNG_MIN) / self.LNG_GRID
        LAT_STEP = (self.LAT_MAX - self.LAT_MIN) / self.LAT_GRID
        lat_centers = np.zeros(self.CNT_GRID)
        lng_centers = np.zeros(self.CNT_GRID)

        # left to right, bottom to top
        LAT_CENTER = self.LAT_MIN - LAT_STEP / 2
        for i in range(self.LAT_GRID):
            LAT_CENTER += LAT_STEP
            LNG_CENTER = self.LNG_MIN - LNG_STEP / 2
            for j in range(self.LNG_GRID):
                LNG_CENTER += LNG_STEP
                grid_idx = self.get_index_from_rowcol(i, j)
                lat_centers[grid_idx], lng_centers[grid_idx] = self.coord_convert(LAT_CENTER, LNG_CENTER)

        for si in range(self.LAT_GRID):
            for sj in range(self.LNG_GRID):
                for di in range(self.LAT_GRID):
                    for dj in range(self.LNG_GRID):
                        # grid (si, sj) to (di, dj)
                        sidx = self.get_index_from_rowcol(si, sj)
                        didx = self.get_index_from_rowcol(di, dj)
                        if sidx == didx:
                            self.distance[sidx][didx] = 0.0
                        elif sidx > didx:
                            self.distance[sidx][didx] = self.distance[didx][sidx]
                        else:
                            self.distance[sidx][didx] = self.path_find(lat_centers[sidx], lng_centers[sidx], lat_centers[didx], lng_centers[didx])
                            print(sidx, didx, self.distance[sidx][didx])
        
        np.save(self.FILE_NAME, self.distance)

dist_crawler = Distance()
dist_crawler.generate_distance()