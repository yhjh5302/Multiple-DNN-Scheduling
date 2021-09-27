import simpy
import numpy as np
import math
from collections.abc import Iterable


class CircleBound:
    def __init__(self, x=0., y=0., boundary=100):
        self.x = x
        self.y = y
        self.sqr_boundary = boundary ** 2
        self.boundary = boundary

    def boundary_chk(self, x, y):
        return (x ** 2) + (y ** 2) <= self.sqr_boundary


class User:
    def __init__(self, idx):
        self.x = None
        self.y = None
        self.num_item, self.popularity = None, None
        self.request_rate = None
        self.result = None
        self.idx = idx
        self.regions = None
        self.unit_time = 1
        self.boundary = None

    def set_boundary(self, boundary):
        self.boundary = boundary

    def set_coordi(self, x, y):
        self.x = x
        self.y = y

    def set_result(self, result):
        self.result = result

    def set_regions(self, regions):
        self.regions = regions

    def set_popularity(self, pop, request_rate):
        self.popularity = pop
        self.num_item = len(self.popularity)
        self.set_request_rate(request_rate)

    def random_move(self, speed, unit_time):
        max_move_radius = speed * unit_time
        max_move_radius = 2 * self.boundary.boundary if max_move_radius > 2 * self.boundary.boundary else max_move_radius
        while True:
            move_radius = np.random.rand() * max_move_radius
            theta = np.random.rand() * 2 * math.pi

            new_x = self.x + move_radius * math.cos(theta)
            new_y = self.y + move_radius * math.sin(theta)
            if self.boundary is None or self.boundary.boundary_chk(new_x, new_y):
                self.x = new_x
                self.y = new_y
                break

    def set_request_rate(self, request_rate):
        self.request_rate = request_rate

    def generating_request(self):
        return np.random.choice(self.num_item, p=self.popularity)

    def request_event(self, time, unit_time):
        request_num = np.random.poisson(self.request_rate)
        if request_num > 0:
            interval = unit_time / request_num
            time += (interval / 2)
            request_chk = False
            for region in self.regions:
                request_chk = region.radius_chk(self)
                if request_chk:
                    break
            if request_chk:
                for i in range(request_num):
                    item = self.generating_request()
                    if not self.result or self.result[-1][0] <= time:
                        self.result.append((time, item))
                    else:
                        insert_point = -1
                        for _ in range(len(self.result) - 1):
                            if self.result[insert_point - 1][0] <= time:
                                break
                            else:
                                insert_point -= 1
                        if insert_point == -len(self.result):
                            self.result = [(time, item)] + self.result
                        else:
                            self.result = self.result[:insert_point] + [(time, item)] + self.result[insert_point:]
                    time += interval
            # if request_num == 0:
            #     yield env.timeout(self.unit_time)
            #     pass
            # else:
            #     interval = 1 / request_num
            #     item = self.generating_request()
            #     yield env.timeout(interval, item)
            #     request_chk = False
            #     for region in self.regions:
            #         request_chk = region.radius_chk(self)
            #         if request_chk:
            #             break
            #
            #     if request_chk and type(self.result) == list:
            #         self.result.append((env.now, item))


class Region:
    def __init__(self):
        self.x = None
        self.y = None
        self.radius = None

    def set_env(self, x, y, rad):
        self.x = x
        self.y = y
        self.radius = rad

    def distance(self, *args):
        if len(args) == 2:
            return math.sqrt((self.x - args[0]) ** 2 + (self.y - args[1]) ** 2)
        else:
            return math.sqrt((self.x - args[0]._x) ** 2 + (self.y - args[0]._y) ** 2)

    def radius_chk(self, *args):
        dist = self.distance(*args)
        return dist <= self.radius


class RequestSim:
    def __init__(self, unit_time):
        self.env = None
        self.regions = None
        self.users = None
        self.unit_time = unit_time
        self.speed = 5
        self.result = None

    def init(self, users, regions):
        self.env = simpy.Environment()
        self.regions = regions
        self.users = users
        self.result = list()
        for user in users:
            user.set_result(self.result)
            user.set_regions(self.regions)

    def reset(self):
        self.env = simpy.Environment()
        self.result = list()

        for user in self.users:
            user.set_result(self.result)

    def unit_time_update(self):
        while True:
            yield self.env.timeout(self.unit_time)
            for user in self.users:
                user.random_move(3, self.unit_time)
                user.request_event(self.env.now, self.unit_time)

    def sim_start(self):
        self.env.process(self.unit_time_update())
        # for user in self.users:
        #     self.env.process(user.request_event(self.env))


if __name__ == "__main__":
    r_sim = RequestSim(1)
    users = [User(i) for i in range(10)]
    regions = [Region() for _ in range(1)]
    idx = 0
    boundary = CircleBound(0., 0., 200)
    regions[0].set_env(0., 0., 100)
    for user in users:
        user.set_popularity([0.1, 0.2, 0.3, 0.4], 10)
        user.set_coordi(0, 0)
        user.set_boundary(boundary)

    r_sim.init(users, regions)
    r_sim.sim_start()
    r_sim.env.run(until=600)
    print(r_sim.result)
