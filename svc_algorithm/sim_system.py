from collections import deque, Counter
import simpy
import numpy as np
import pickle
from svc_algorithm import requester
import math


class Server:
    def __init__(self, env, cpu, mem, cp):
        self.env = env
        self.cpu = cpu
        self.cp = cp
        self.mem = mem
        self.pushed_lst = list()

    def reset(self, env):
        self.env = env
        self.pushed_lst.clear()

    def set_bandwidth(self, bandwidth):
        pass

    def communication_time(self, destination, data):
        pass

    def computing_time(self, amount):
        pass


class Edge(Server):
    def __init__(self, env, cpu, mem, cp, edge_id):
        super().__init__(env, cpu, mem, cp)
        self.id = edge_id
        self.bandwidth_arr = None
        self.cluster = None
        self.K = None
        self.usage = None

    def link_cluster(self, cluster):
        self.cluster = cluster
        self.K = cluster.K

    def set_bandwidth(self, array):
        self.bandwidth_arr = array

    def communication_time(self, destination, data):
        return data / self.bandwidth_arr[destination]

    def computing_time(self, amount):
        return amount / self.cp

    def cal_resource_usage(self):
        pushed_lst = self.K[self.id, :]
        pushed_lst = pushed_lst.T
        self.usage = np.sum(pushed_lst * self.cluster.require_map, axis=0)
        return self.usage

    def capacity_check(self, cpu_req, mem_req):  # true : can push, false: can not push
        usage_chk = self.usage + np.array((cpu_req, mem_req))
        return not np.any(usage_chk > np.array(self.cpu, self.mem))


class EdgeCluster:
    def __init__(self, edge_lst, svc_set):
        self.edge_lst = edge_lst
        self.svc_set = svc_set
        self.num_edge = len(edge_lst)
        self.num_service = svc_set.num_service
        self.num_micro_service = svc_set.num_micro_service
        self.micro_service_map = list()
        self.require_map = svc_set.require_map
        self.resource_map = np.zeros((self.num_edge, 2))
        self.avail_map = np.zeros((self.num_edge, 2))
        self.K = np.zeros((self.num_edge, self.num_micro_service), dtype=np.bool_)

        for edge in self.edge_lst:
            edge.link_cluster(self)

    def reset(self, env):
        for edge in self.edge_lst:
            edge.reset(env)
        self.K = np.zeros((self.num_edge, self.num_micro_service), dtype=np.bool_)

    def find_candidate_edge(self, *args):
        if len(args) == 1:
            k_idx = args[0]
        else:
            k_idx = self.svc_set.find_k_idx(args[0], args[1])

        cpu_require = self.require_map[0, k_idx]
        mem_require = self.require_map[1, k_idx]

        result = np.logical_and(self.avail_map[:, 0] > cpu_require, self.avail_map[:, 1] > mem_require)
        return np.logical_and(result, np.logical_not(self.K[:, k_idx]))

    def update_usage(self, edge_idx):
        usage = self.edge_lst[edge_idx].cal_resource_usage()
        self.avail_map[edge_idx, 0] = self.resource_map[edge_idx, 0] - usage[0]
        self.avail_map[edge_idx, 1] = self.resource_map[edge_idx, 1] - usage[1]
        if np.any(self.avail_map < 0):
            raise Exception("Violation  of restrictions: constraint ")

    def push_service_to_edge(self, edge_idx, service_idx, micro_service_idx):
        k_idx = self.svc_set.find_k_idx(service_idx, micro_service_idx)
        ms = self.svc_set.svc_lst[service_idx].MS_lst[micro_service_idx]
        if self.edge_lst[edge_idx].capacity_check(ms.cpu_req, ms.mem_req):
            self.K[:, k_idx] = False
            self.K[edge_idx, k_idx] = True
            self.update_usage(edge_idx)

    def set_k(self, K):
        for edge_idx in range(self.num_edge):
            self.K[edge_idx, :] = K[edge_idx, :]
            self.update_usage(edge_idx)

    def find_pushed_server(self, k_idx):
        if np.any(self.K[:, k_idx] == 1):
            result = np.where(self.K[:, k_idx] == 1)
            return result[0][0]
        else:
            return -1

    def constraint_chk(self, new_K):  # (batch, container)
        cpu_need = new_K * self.require_map[0, :].reshape((1, -1))
        cpu_need = np.sum(cpu_need, axis=1)
        cpu_need = cpu_need.reshape((-1, 1))

        mem_need = new_K * self.require_map[1, :].reshape(1,-1)
        mem_need = np.sum(mem_need, axis=1)
        mem_need = mem_need.reshape((-1, 1))

        return np.logical_and(cpu_need <= self.resource_map[:, 0].reshape((1, -1)),
                              mem_need <= self.resource_map[:, 1].reshape((1, -1)))


class Cloud(Server):
    def __init__(self, env, cp):
        super().__init__(env, cpu=float("inf"), mem=float("inf"), cp=cp)
        self.bandwidth = None

    def set_bandwidth(self, bandwidth):
        self.bandwidth = bandwidth

    def communication_time(self, destination, data):
        return data / self.bandwidth

    def computing_time(self, amount):
        return amount / self.cp


class MicroService:
    def __init__(self, cpu_req, mem_req, output_size, amount, idx):
        self.cpu_req = cpu_req
        self.mem_req = mem_req
        self.output_size = output_size
        self.amount = amount
        self.idx = idx
        self.parents = set()
        self.children = set()

    def add_parent(self, parent, inter_opt=True):
        self.parents.add(parent)
        if inter_opt:
            parent.add_child(self, False)

    def add_child(self, child, inter_opt=True):
        self.children.add(child)
        if inter_opt:
            child.add_parent(self, False)


class MicroServiceSet:
    def __init__(self, input_size, deadline):
        self.MS_lst = list()
        self.start = None
        self.end = None
        self.first_MS = 0
        self.input_size = input_size
        self.num_micro_service = 0
        self.deadline = deadline

    def set_seq_micro_service(self, micro_lst):
        self.num_micro_service = len(micro_lst)
        self.MS_lst = micro_lst
        for ms_idx in range(self.num_micro_service - 1):
            self.MS_lst[ms_idx].add_child(self.MS_lst[ms_idx+1])
        self.start = 0
        self.end = self.num_micro_service - 1

    def get_input_size(self, ms_idx):
        if ms_idx == 0:
            return self.input_size
        else:
            ms = self.MS_lst[ms_idx]
            input_size = 0
            for parent in ms.parents:
                input_size += parent.output_size


class ServiceSet:
    def __init__(self, svc_lst):
        self.svc_lst = svc_lst
        self.num_service = len(svc_lst)
        self.num_micro_service = 0
        self.micro_service_map = list()
        self.micro_service_map_2 = np.empty(1, dtype=np.int_)
        svc_idx = 0
        for svc in svc_lst:
            start = self.num_micro_service
            self.num_micro_service += svc.num_micro_service
            end = self.num_micro_service - 1
            self.micro_service_map.append((start, end))
            msvc2svc = np.full(svc.num_micro_service, svc_idx)
            self.micro_service_map_2 = np.append(self.micro_service_map_2, msvc2svc)
            svc_idx += 1

        self.micro_service_map_2 = np.delete(self.micro_service_map_2, [0])

        self.require_map = np.zeros((2, self.num_micro_service))
        k_idx = 0
        for svc_idx in range(self.num_service):
            start, end = self.micro_service_map[svc_idx]
            for i in range(start, end + 1):
                self.require_map[0, k_idx] = self.svc_lst[svc_idx].MS_lst[i - start].cpu_req
                self.require_map[1, k_idx] = self.svc_lst[svc_idx].MS_lst[i - start].mem_req
                k_idx += 1

    def find_k_idx(self, service_idx, micro_service_idx):
        return self.micro_service_map[service_idx][0] + micro_service_idx

    def find_svc_idx(self, k_idx):
        return self.micro_service_map_2[k_idx]


class User:
    def __init__(self, edge_bandwidth, cloud_bandwidth):
        self.edge_bandwidth = edge_bandwidth
        self.cloud_bandwidth = cloud_bandwidth

    def communication_time(self, destination, data):
        if destination == -1:
            return data / self.cloud_bandwidth

        elif type(self.edge_bandwidth) in (int, float):
            return data / self.edge_bandwidth

        else:
            return data / self.edge_bandwidth[destination]


class Data:
    def __init__(self, service_id, dnn_order,size):
        self.service_id = service_id
        self.order = dnn_order
        self.size = size


class DataTransfer: # calculate transport time
    def __init__(self):
        self.user = None
        self.edge_cluster = None
        self.cloud = None

    def set_element(self, user, edge_cluster, cloud):
        self.user = user
        self.edge_cluster = edge_cluster
        self.cloud = cloud

    def transfer(self, source, destination, data):  # 0>=: edge, -1: cloud, -2: user
        if type(data) == Data:
            data = data.size

        if source == destination:
            return 0

        elif source == -2 or destination == -2:
            target = source if destination == -2 else destination
            return self.user.communication_time(target, data)

        elif source == -1 or destination == -1:
            target = source if destination == -1 else destination
            return self.cloud.communication_time(target, data)

        else:
            self.edge_cluster.edge_lst[source].communication_time(destination, data)


class MicroServiceEvent:
    def __init__(self, env, ms, place, sim):
        self.idx = ms.idx
        self.env = env
        self.MS = ms
        self.place_idx = place
        self.place = sim.edge_cluster[place] if place >= 0 else sim.cloud
        self.input_events = {parent.idx: env.event() for parent in ms.parents}
        self.output_events = list()
        self.sim = sim

    def link_with_child(self, child):
        self.output_events.append(child.input_events[self.idx])

    def set_first_event(self, event):
        self.input_events[-1] = event

    def set_last_event(self):
        # t_tr = self.sim.transfer(self.place_idx, -2, self.MS.output_size)
        # self.output_events.append(self.env.timeout(t_tr))
        self.output_events.append(self.env.event())

    def processing(self):
        yield simpy.AllOf(env=self.env, events=self.input_events.values())  # wait all inputs arrive
        # print("processing %s start %s" % (self.idx, self.env.now))
        yield self.env.timeout(self.place.complete_time(self.MS.amount))  # computing
        # print("processing %s end %s" % (self.idx, self.env.now))

        if self.MS.children:
            t_tr = [self.sim.transfer(self.place_idx, child.idx, self.MS.output_size) for child in self.MS.children]
            diff_arr = np.array(t_tr)
            min_delay = diff_arr.min()
            diff_arr = diff_arr - min_delay
            yield self.env.timeout(min_delay)

            for i in diff_arr.argsort():
                yield self.env.timeout(t_tr[i])
                self.output_events[i].succeed()

        else:
            t_tr = self.sim.transfer(self.place_idx, -2, self.MS.output_size)
            yield self.env.timeout(t_tr)
            self.output_events[0].succeed()


class Simulator(DataTransfer):  # processing micro service set
    def __init__(self):
        super().__init__()
        self.svc_set = None
        self.env = None
        self.result = None
        self.state = None

    def set_env(self, env, svc_set, user, cloud, edge_cluster):
        self.env = env
        self.set_element(user, edge_cluster, cloud)
        self.svc_set = svc_set
        self.result = list()
        self.state = np.zeros(edge_cluster.num_micro_service, dtype=np.int_)

    def reset(self, env):
        self.cloud.reset(env)
        self.edge_cluster.reset(env)
        self.result = list()
        self.state = np.zeros_like(self.state)

    def request_event(self, service_idx, request_time):
        result = dict()
        result['service_idx'] = service_idx
        yield self.env.timeout(request_time)  # wait until request occurred
        result['start'] = self.env.now  # request start
        # print("request %s: start in %s" % (service_idx, self.env.now))
        svc = self.svc_set.svc_lst[service_idx]
        events = list()
        for ms_idx in range(len(svc.MS_lst)):
            k_idx = self.svc_set.find_k_idx(service_idx, ms_idx)
            self.state[k_idx] = self.state[k_idx] + 1  # add request event
            place = self.edge_cluster.find_pushed_server(k_idx)
            events.append(MicroServiceEvent(self.env, svc.MS_lst[ms_idx], place, self))

        for event in events:
            for child in event.MS.children:
                event.link_with_child(events[child.idx])
        events[-1].set_last_event()
        first_event = self.env.event()
        events[svc.first_MS].set_first_event(first_event)
        for event in events:
            self.env.process(event.processing())

        k_idx = self.svc_set.find_k_idx(service_idx, svc.first_MS)
        place = self.edge_cluster.find_pushed_server(k_idx)
        t_tr = self.transfer(-2, place, svc.input_size)

        yield self.env.timeout(t_tr)  # transfer time
        first_event.succeed()
        # print("request %s: first in %s" % (service_idx, self.env.now))

        yield events[-1].output_events[0]
        # print("request %s: end in %s" % (service_idx, self.env.now))
        result['end'] = self.env.now  # request_end
        result['success'] = True
        self.result.append(result)
        # return result

    def run_request_event(self, service_idx, request_time):
        self.env.process(self.request_event(service_idx, request_time))
        # self.result.append(result)
        # return result


class RequestGenerator:
    def __init__(self, request_info, end_t):
        self.time, self.end_t, self.request_info, self.num_svc = None, None, None, None
        self.r_info_idx = None
        self.set_end_t(end_t)
        self.set_request_info(request_info)

    def set_end_t(self, end_t):
        self.end_t = end_t

    def set_request_info(self, request_info):  # end_t(seconds), request rate
        self.request_info = request_info
        self.request_info = sorted(self.request_info, key=lambda x: x[0])

        if type(self.request_info[0][1]) == np.ndarray:
            self.num_svc = self.request_info[0][1].size
        else:
            self.num_svc = 0

        if self.request_info[-1][0] < self.end_t:
            self.request_info.append((self.end_t, 0.0))

    def chk_r_info_idx(self):
        if self.request_info is not None:
            if self.r_info_idx > 0:
                start_t = self.request_info[self.r_info_idx - 1][0]
            else:
                start_t = 0
            if self.r_info_idx < len(self.request_info):
                return start_t <= self.time < self.request_info[self.r_info_idx][0]
            else:
                return True
        else:
            return False

    def find_r_info_idx(self, step_by_step=True):
        ex_r_info_idx = self.r_info_idx
        if step_by_step:
            if self.r_info_idx is None:
                self.r_info_idx = 0
            while not self.chk_r_info_idx() and self.r_info_idx < len(self.request_info) - 1:
                self.r_info_idx += 1

        else:  # using quick sort
            search_range = [0, len(self.request_info)]
            while True:
                self.r_info_idx = (search_range[0] + search_range[1]) / 2
                if self.chk_r_info_idx():
                    break
                else:
                    if self.r_info_idx == 0 or self.request_info[self.r_info_idx - 1][0] > self.time:
                        search_range = [search_range[0], self.r_info_idx - 1]
                    else:
                        search_range = [self.r_info_idx + 1, search_range[1]]
        return ex_r_info_idx != self.r_info_idx

    def reset(self):
        self.time = 0.0
        self.r_info_idx = None

    def step(self, period):
        request_arr = np.zeros(self.num_svc, dtype=np.int_)
        period_end_t = self.time + period
        while self.time < period_end_t:
            r_period = self.request_info[self.r_info_idx][0]  # request end time
            if r_period > period_end_t:
                r_period = period_end_t
            request_arr += np.random.poisson(self.request_info[self.r_info_idx][1] * (r_period - self.time))
            self.time = r_period
            self.find_r_info_idx()

        return self.time >= self.end_t, request_arr

    def load_file(self, file_path):
        with open(file_path, 'rb') as f:
            load_data = pickle.load(f)
        # self.end_t, self.request_info, self.num_svc = load_data.end_t, load_data.request_info, load_data.num_svc
        for key in load_data.__dict__.keys():
            self.__dict__[key] = load_data.__dict__[key]
        self.reset()

    def save_file(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)


class RequestGenerator2(RequestGenerator):
    def __init__(self, request_info, end_t, num_svc, num_user=100, comm_radius=100., sim_bound=100.*1.414, unit_time=5 * 60):
        super(RequestGenerator2, self).__init__(request_info, end_t)
        self.num_svc = num_svc
        self.model = requester.RequestSim(unit_time=unit_time)
        users = [requester.User(i) for i in range(num_user)]
        regions = [requester.Region() for _ in range(1)]
        boundary = requester.CircleBound(0., 0., sim_bound)
        regions[0].set_env(0., 0., comm_radius)
        self.high_user, self.low_user = list(), list()

        for user in users:
            pop = np.random.rand(self.num_svc)
            pop = pop / pop.sum()

            user.set_popularity(pop, 1)
            user.set_boundary(boundary)
            if np.random.random() < 0.2:
                self.high_user.append(user)
            else:
                self.low_user.append(user)

            while True:
                r, theta = np.random.rand(2)
                r *= sim_bound
                theta *= 2 * math.pi
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                if boundary.boundary_chk(x, y):
                    break
            user.set_coordi(x, y)
        self.model.init(users, regions)

        self.using_record = False
        self.records, self.record_idx, self.record_epi_idx, self.epi_end = None, None, None, None
        self.pop = None

    def set_using_record(self, opt):  # using record data
        self.using_record = opt

    def update_user_request(self):
        if self.find_r_info_idx():
            period = self.request_info[self.r_info_idx][0] - self.request_info[self.r_info_idx - 1][0] if self.r_info_idx > 0 else self.request_info[self.r_info_idx][0]
            period = float(period)
            high_rate = self.request_info[self.r_info_idx][1] * 0.8 / len(self.high_user) / period * self.model.unit_time
            low_rate = self.request_info[self.r_info_idx][1] * 0.2 / len(self.low_user) / period * self.model.unit_time

            for user in self.high_user:
                user.set_request_rate(high_rate)

            for user in self.low_user:
                user.set_request_rate(low_rate)

    def make_random_record(self, num_records, period):
        num_epi = math.ceil(self.end_t / period)
        self.epi_end = num_epi
        self.records = np.zeros((num_records, num_epi, self.num_svc), dtype=np.int_)
        self.pop = np.zeros((num_records, self.num_svc), dtype=np.float_)
        for record_idx in range(num_records):
            self.model.reset()
            self.model.sim_start()
            epi_idx = 0
            self.time = 0
            while self.time < self.end_t:
                period_end_t = self.time + period
                while self.time < period_end_t:
                    self.update_user_request()
                    self.model.env.run(until=self.time + self.model.unit_time)
                    self.time = self.model.env.now

                while self.model.result:
                    r = self.model.result.pop(0)
                    self.records[record_idx][epi_idx][r[1]] += 1
                    self.pop[record_idx][r[1]] += 1
                epi_idx += 1
            self.pop[record_idx] = self.pop[record_idx] / self.pop[record_idx].sum()

    def get_pop(self):
        if self.using_record:
            return self.pop[self.record_idx]
        else:
            if self.pop is None:
                self.step(60 * 60) # todo: change period option
            return self.pop

    def reset(self):
        super(RequestGenerator2, self).reset()
        if self.using_record:
            self.record_epi_idx = 0
            self.record_idx = 0 if self.record_idx is None else (self.record_idx + 1) % len(self.records)
        else:
            self.model.reset()
            self.model.sim_start()
            self.records, self.pop = None, None

    def step(self, period):
        if self.using_record:
            if self.record_epi_idx < self.epi_end:
                result = self.records[self.record_idx][self.record_epi_idx]
                self.record_epi_idx += 1
                return self.record_epi_idx > self.epi_end, result
            else:
                return True, np.zeros(self.num_svc)

        else:
            if self.records is None:
                self.records = list()
                self.pop = np.zeros(self.num_svc, dtype=np.float_)
                done = False
                while not done:
                    period_end_t = self.time + period
                    while self.time < period_end_t:
                        self.update_user_request()
                        self.model.env.run(until=self.time + self.model.unit_time)
                        self.time = self.model.env.now

                    result = np.zeros(self.num_svc, dtype=np.int_)
                    while self.model.result:
                        r = self.model.result.pop(0)
                        result[r[1]] += 1
                        self.pop[r[1]] += 1
                    done = self.time >= self.end_t
                    self.records.append((done, result))
                self.time = 0
                self.pop = self.pop / self.pop.sum()
            self.time += period

            return self.records.pop(0)  # pop first item (fifo)

    def save_file(self, file_path):
        with open(file_path, 'wb') as f:
            model = self.model
            self.model = None
            pickle.dump(self, f)
            self.model = model


class SimulatorPack:
    def __init__(self, num_edge=10, num_service=10, end_t=60*60*24, alpha=2.):
        self.env = simpy.Environment()
        self.sim = Simulator()
        self.num_edge = num_edge
        self.num_service = num_service
        self.edge_bandwidth = np.full(10, 100 * (1000 ** 2))  # 100Mbps
        self.cloud_bandwidth = 10 * (1000 ** 2)  # 10Mbps
        self.cloud_cp = 1000
        self.edge_cp = 100
        self.end_t = end_t
        self.alpha = alpha

        self.user = User(self.edge_bandwidth, self.cloud_bandwidth)
        self.cloud = Cloud(self.env, self.cloud_cp)
        self.cloud.set_bandwidth(self.cloud_bandwidth)
        self.edge_lst = [Edge(self.env, 100, 8 * (1024 ** 3), self.edge_cp, idx) for idx in range(self.num_edge)]
        for edge in self.edge_lst:
            edge.set_bandwidth(np.full((self.num_edge, self.num_edge), self.edge_bandwidth))

        svc_lst = list()
        for service_idx in range(self.num_service):
            ms_lst = list()
            for layer_idx in range(np.random.randint(4, 11)):
                ms_lst.append(MicroService(10, 50 * (1024 ** 2), 10 * 1024, 10, layer_idx))
            ms_set = MicroServiceSet(10 * (1024 ** 2), 0.001)
            ms_set.set_seq_micro_service(ms_lst)
            svc_lst.append(ms_set)
        self.svc_set = ServiceSet(svc_lst)
        self.edge_cluster = EdgeCluster(self.edge_lst, self.svc_set)
        self.sim.set_env(self.env, self.svc_set, self.user, self.cloud, self.edge_cluster)
        self.constraint_chk = True


    def reset(self):
        self.env = simpy.Environment()
        self.sim.reset(self.env)
        self.result_lst = list()

    def get_input_size(self):
        return self.edge_cluster.resource_map.size + self.edge_cluster.require_map.size + \
               self.edge_cluster.num_micro_service

    def get_output_size(self):
        return self.edge_cluster.K.size

    def get_state(self):
        cpu_map = np.concatenate((self.edge_cluster.resource_map[:, 0], self.svc_set.require_map[0, :]))
        mem_map = np.concatenate((self.edge_cluster.resource_map[:, 1], self.svc_set.require_map[1, :]))
        if np.max(cpu_map) != 0:
            cpu_map = cpu_map / np.max(cpu_map)
        if np.max(mem_map) != 0:
            mem_map = mem_map / np.max(mem_map)
        result = np.concatenate((cpu_map, mem_map))
        result = np.concatenate((result, self.sim.state))
        return result

    def step(self, period=60):
        self.sim.result.clear()
        if self.env.now + period >= self.end_t:
            done = True
        else:
            done = False

        self.env.run(until=self.env.now + period)
        # self.result_lst.append(self.sim.result)
        self.result_lst = self.result_lst + self.sim.result
        return self.reward(self.sim.result), done

    def set_k(self, K):
        K = np.reshape(K, (self.num_edge, self.edge_cluster.num_micro_service))
        self.constraint_chk = self.edge_cluster.constraint_chk(K).all()
        if self.constraint_chk:
            self.edge_cluster.set_k(K)

    def reward(self, result_lst):
        result = 0.
        if self.constraint_chk:
            for result in result_lst:
                service_idx = result['service_idx']
                delay = result['end'] - result['start']
                time_limit = self.svc_set.svc_lst[service_idx].deadline
                if delay <= time_limit:
                    result += 1.0
                elif delay > self.alpha * time_limit:
                    result += 0.0
                else:
                    result += ((self.alpha * time_limit - delay) / ((self.alpha - 1.0) * time_limit))
        return result


class SimpleSimulator:
    def __init__(self, db_conn=None, request_gen=None):
        self.db_conn = db_conn
        self.node_info = None
        self.msvc_data, self.require_map, self.svc_map, self.svc_info = None, None, None, None
        self.deadline, self.fog_time, self.cloud_time, self.cloud_trans_time = None, None, None, None
        self.resource_map = None
        self.cpu_require = None
        self.mem_require = None
        self.request_gen = request_gen
        self.mat_k = None
        self.alpha = 2.0
        self.episode_period = None
        self.state = None
        self.ex_reward = None
        self.time_ratio = None
        self.svc_map_mat = None
        if db_conn is not None:
            self.db_init()

    def set_episode_period(self, episode_period):
        self.episode_period = episode_period

    def db_init(self):
        self.node_info, self.resource_map = self.db_conn.get_node_info()
        self.msvc_data, self.require_map, self.svc_map = self.db_conn.get_msvc_data()
        self.svc_info = self.db_conn.get_svc_data()
        self.cpu_require = self.require_map[0, :]
        self.cpu_require = self.cpu_require.reshape((1, -1))
        self.cpu_require = self.cpu_require.repeat(len(self.node_info), axis=0)
        self.mem_require = self.require_map[1, :]
        self.mem_require = self.mem_require.reshape((1, -1))
        self.mem_require = self.mem_require.repeat(len(self.node_info), axis=0)
        self.time_ratio = np.zeros((1, len(self.msvc_data)), dtype=np.float_)
        self.svc_map_mat = np.zeros((len(self.svc_map), len(self.msvc_data)), dtype=np.bool_)

        for svc_key in self.svc_map:
            msvc_idx = self.key_to_idx(self.svc_map[svc_key])
            self.svc_map_mat[self.key_to_idx(svc_key), msvc_idx] = True
            self.time_ratio[0, msvc_idx] = self.svc_info[svc_key][1]   # deadline
            for m_i in msvc_idx:
                self.time_ratio[0, m_i] = self.msvc_data[self.idx_to_key(m_i)][1] / self.time_ratio[0, m_i]

    @staticmethod
    def key_to_idx(key):
        return key - 1  # todo change

    @staticmethod
    def idx_to_key(idx):
        return idx + 1

    def set_action(self, action):
        # result = np.zeros((1, self.mat_k.size), dtype=np.bool_)
        # result[0, action[0, 0]] = True
        # result = result.reshape(len(self.node_info), len(self.msvc_data))
        # result = np.logical_or(self.mat_k, result)
        # self.set_k(result)
        mat_k = self.mat_k.ravel()
        changed = False
        if not mat_k[action[0, 0]]:
            mat_k[action[0, 0]] = True
            changed = True
        constraint_chk = self.constraint_chk(self.mat_k)

        if not constraint_chk:
            mat_k[action[0, 0]] = False
        return constraint_chk, changed

    def set_k(self, mat_k):
        if self.constraint_chk(mat_k):
            self.mat_k = mat_k
            return True
        else:
            return False

    def req_step(self, period=None):
        if period is None:
            period = self.episode_period

        r_done, arr_request = self.request_gen.step(period)
        r_sum = arr_request.sum()
        if r_sum != 0:
            arr_request = arr_request / arr_request.sum()  # normalize

        for svc_key in self.svc_map:
            self.state[self.key_to_idx(self.svc_map[svc_key])] = arr_request[self.key_to_idx(svc_key)]

        return arr_request, r_done

    def step(self, arr_request, constraint_chk, changed):
        if not constraint_chk:
            done = True
        elif not changed:
            done = True
        else:
            not_in_fog = np.logical_not(np.any(self.mat_k, axis=0))
            if np.any(not_in_fog):
                cpu_req = self.cpu_require * not_in_fog
                mem_req = self.mem_require * not_in_fog

                avail = self.avail_map(self.mat_k)
                usable_node = np.logical_and(
                    avail[:, 0] > np.min(cpu_req[np.nonzero(cpu_req)]),
                    avail[:, 1] > np.min(mem_req[np.nonzero(mem_req)])
                )
                if np.any(usable_node):  #
                    done = False
                else:
                    done = True

            else:  # all service is stored
                done = True
            # done = False
        cur_eval = self.evaluate(self.mat_k, arr_request, self.alpha)
        if self.ex_reward is None:
            result = cur_eval

        else:
            result = cur_eval - self.ex_reward
        self.ex_reward = cur_eval
        return result, done

    def req_gen_reset(self):
        self.request_gen.reset()
        self.state_reset()

    def state_reset(self):
        self.state = np.zeros(len(self.msvc_data), dtype=np.float_)
        self.mat_k = np.zeros((len(self.node_info), len(self.msvc_data)))
        self.ex_reward = None

    def resource_usage(self, mat_k):
        cpu_r = mat_k * self.cpu_require
        mem_r = mat_k * self.mem_require

        cpu_r = cpu_r.sum(axis=1)
        mem_r = mem_r.sum(axis=1)
        return cpu_r, mem_r

    def avail_map(self, mat_k):
        result = np.zeros_like(self.resource_map)
        cpu_r, mem_r = self.resource_usage(mat_k)
        result[:, 0] = self.resource_map[:, 0] - cpu_r
        result[:, 1] = self.resource_map[:, 1] - mem_r
        return result

    def constraint_chk(self, mat_k, result_arr=False):
        cpu_r, mem_r = self.resource_usage(mat_k)
        if result_arr:
            return np.logical_and(cpu_r <= self.resource_map[:, 0], mem_r <= self.resource_map[:, 1])
        else:
            return np.all(cpu_r <= self.resource_map[:, 0]) and np.all(mem_r <= self.resource_map[:, 1])

    def get_input_size(self):
        return len(self.node_info) * 2 + len(self.msvc_data) + len(self.msvc_data) * 2 + \
               len(self.msvc_data) * len(self.node_info)

    def get_output_size(self):
        return len(self.node_info) * len(self.msvc_data)

    def get_state(self):
        avail_map = self.avail_map(self.mat_k)
        # usable = avail_map[:, 0].reshape((-1, 1)) - self.require_map[0, :].reshape((1, -1))
        # mask = usable > 0.
        # usable = avail_map[:, 1].reshape((-1, 1)) - self.require_map[1, :].reshape((1, -1))
        # mask = np.logical_and(mask, usable > 0.)
        # in_fog = np.any(self.mat_k, axis=0, keepdims=True)
        # local_search = np.zeros_like(in_fog)

        # for svc_key in self.svc_map:
        #     for msvc_idx in self.key_to_idx(self.svc_map[svc_key]):
        #         if not in_fog[0, msvc_idx]:  # first not in fog
        #             local_search[0, msvc_idx] = True
        #             break
        # mask = mask * local_search
        # mask = self.time_ratio * mask
        cpu_map = np.concatenate((avail_map[:, 0], self.require_map[0, :]))
        mem_map = np.concatenate((avail_map[:, 1], self.require_map[1, :]))

        if np.max(cpu_map) != 0:
            cpu_map = cpu_map / max(np.max(cpu_map), np.max(self.resource_map[:, 0]))
            # cpu_map = cpu_map / np.max(cpu_map)

        if np.max(mem_map) != 0:
            mem_map = mem_map / max(np.max(mem_map), np.max(self.resource_map[:, 1]))
            # mem_map = mem_map / np.max(mem_map)

        result = np.concatenate((cpu_map, mem_map, self.mat_k.flatten(), self.state))
        return result

    def evaluate(self, mat_k, arr_request, alpha):  # convert to reward
        result = 0.0
        mat_fog = mat_k.sum(axis=0)
        for svc_key in self.svc_map:
            deadline = self.svc_info[svc_key][1]
            tr_time = self.svc_info[svc_key][2]

            total_time = 0.0
            in_fog = True
            for msvc_key in self.svc_map[svc_key]:
                if not mat_fog[self.key_to_idx(msvc_key)] and in_fog:
                    total_time += tr_time
                    in_fog = False
                if mat_fog[self.key_to_idx(msvc_key)]:
                    total_time += self.msvc_data[msvc_key][1]
                else:
                    total_time += self.msvc_data[msvc_key][2]

            if total_time < deadline:
                result += arr_request[self.key_to_idx(svc_key)]
            elif total_time < alpha * deadline:
                result += arr_request[self.key_to_idx(svc_key)] * ((alpha * deadline - total_time) /
                                                                   ((alpha - 1.0) * deadline))
            else:
                pass  # add zero
        # return result
        return result / np.sum(arr_request) if result != 0. else 0.0   # for normalizing


if __name__ == "__main__":
    env = simpy.Environment()
    sim = Simulator()
    num_edge = 10
    num_service = 10
    edge_bandwidth = np.full(10, 100*(1000**2))  # 100Mbps
    cloud_bandwidth = 10 * (1000 ** 2)  # 10Mbps
    user = User(edge_bandwidth, cloud_bandwidth)
    cloud_cp = 1000
    edge_cp = 100
    cloud = Cloud(env, cloud_cp)
    cloud.set_bandwidth(cloud_bandwidth)
    edge_lst = [Edge(env, 100, 8*(1024**3), edge_cp, idx) for idx in range(num_edge)]

    for edge in edge_lst:
        edge.set_bandwidth(np.full((num_edge, num_edge), edge_bandwidth))
    svc_lst = list()
    for service_idx in range(num_service):
        ms_lst = list()
        for layer_idx in range(np.random.randint(4, 11)):
            ms_lst.append(MicroService(10, 50*(1024**2), 10*1024, 10, layer_idx))
        ms_set = MicroServiceSet(10*(1024**2), 0.001)
        ms_set.set_seq_micro_service(ms_lst)
        svc_lst.append(ms_set)
    svc_set = ServiceSet(svc_lst)
    edge_cluster = EdgeCluster(edge_lst, svc_set)
    sim.set_env(env, svc_set, user, cloud, edge_cluster)
    result_lst = list()
    result_lst.append(sim.run_request_event(0, 10))
    result_lst.append(sim.run_request_event(1, 20))

    env.run(until=11)

    for result in result_lst:
        if 'success' in result and result['success']:
            print("service %s start %s" % (result['service_idx'], result['start']))
            print("service %s end %s" % (result['service_idx'], result['end']))
    env.run(until=20)

    for result in result_lst:
        if 'success' in result and result['success']:
            print("service %s start %s" % (result['service_idx'], result['start']))
            print("service %s end %s" % (result['service_idx'], result['end']))
