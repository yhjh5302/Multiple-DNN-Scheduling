import numpy as np
import random
from dag_server import *


class DAGDataSet:
    def __init__(self, max_timeslot):
        self.max_timeslot = max_timeslot
        self.svc_set, self.system_manager = self.data_gen()

    def create_arrival_rate(self, num_services, minimum, maximum):
        return minimum + (maximum - minimum) * np.random.random(num_services)

    def data_gen(self, num_services=1, max_partitions=20, deadline_opt=(10, 20), num_edges=5, num_fogs=1, num_clouds=1,
                ipc=(10**12), B_gw=1024*40, B_fog=1024*10, B_cl=1024*1, P_dd_opt=(0.5,1)):
        # ipc -> TFLOPS
        # create system manager
        system_manager = SystemManager()

        # create service set
        svc_set = ServiceSet()
        for i in range(num_services):
            # create service
            deadline = random.uniform(deadline_opt[0], deadline_opt[1])
            svc = Service(deadline)

            # create partitions
            dag_shape = (0, ((1, random.randint(2, max_partitions)), (0, random.randint(1, max_partitions)), (1, ((0, random.randint(1, max_partitions)), (0, random.randint(1, max_partitions)))), (0, 1)))
            dag_size = svc.calc_service_size(shape=dag_shape) + 1
            svc.input_data_array = np.zeros(shape=(dag_size, dag_size), dtype=np.int32)
            svc.create_partitions(dag_shape)

            svc_set.add_services(svc)
        self.num_services = len(svc_set.services)
        self.num_partitions = len(svc_set.partitions)

        # create arrival rate table
        self.max_arrival = 50
        self.min_arrival = 10

        svc_arrival = list()
        for t in range(self.max_timeslot):
            svc_arrival.append(self.create_arrival_rate(num_services=num_services, minimum=self.min_arrival, maximum=self.max_arrival))

        # create servers
        self.num_servers = num_edges + num_fogs + num_clouds
        self.num_edges = num_edges
        self.num_fogs = num_fogs
        self.num_clouds = num_clouds
        edge = dict()
        fog = dict()
        cloud = dict()

        device_types = ['tiny', 'small', 'large', 'mobile']
        for i in range(num_edges):
            device = np.random.choice(device_types, p=[0.1, 0.1, 0.4, 0.4])
            if device == 'tiny':
                cpu = random.randint(3, 5) / 1000 # Tflops
                mem = random.randint(2, 4) * 1024 * 1024 # KB
            elif device == 'small':
                cpu = random.randint(10, 20) / 1000 # Tflops
                mem = random.randint(2, 8) * 1024 * 1024 # KB
            elif device == 'large':
                cpu = random.randint(500, 2000) / 1000 # Tflops
                mem = random.randint(4, 16) * 1024 * 1024 # KB
            elif device == 'mobile':
                cpu = random.randint(1000, 2000) / 1000 # Tflops
                mem = random.randint(4, 16) * 1024 * 1024 # KB
            else:
                raise RuntimeError('Unknown device type {}'.format(device))
            edge[i] = Server(cpu, mem, ipc, system_manager=system_manager, id=i)
        for i in range(num_edges, num_edges + num_fogs):
            fog_cpu = random.randint(5, 5) # Tflops
            fog_mem = random.randint(16, 16) * 1024 * 1024 # KB
            fog[i] = Server(fog_cpu, fog_mem, ipc, system_manager=system_manager, id=i)
        for i in range(num_edges + num_fogs, num_edges + num_fogs + num_clouds):
            cloud_cpu = random.randint(30, 30) # Tflops
            cloud_mem = random.randint(256, 256) * 1024 * 1024 # KB
            cloud[i] = Server(cloud_cpu, cloud_mem, ipc, system_manager=system_manager, id=i)

        # create network manager
        noise = 1
        channel_bandwidth = 1024*25
        channel_gain = 1
        net_manager = NetworkManager(channel_bandwidth, channel_gain, noise, system_manager)
        net_manager.B_gw = B_gw
        net_manager.B_fog = B_fog
        net_manager.B_cl = B_cl
        net_manager.P_dd = np.zeros(shape=(self.num_servers, self.num_servers))
        for i in range(self.num_servers):
            for j in range(i + 1, self.num_servers):
                net_manager.P_dd[i, j] = net_manager.P_dd[j, i] = random.uniform(P_dd_opt[0], P_dd_opt[1])
            net_manager.P_dd[i, i] = 0
        net_manager.cal_b_dd()

        # init system manager
        system_manager.net_manager = net_manager
        system_manager.num_servers = self.num_servers
        system_manager.num_services = self.num_services
        system_manager.num_partitions = self.num_partitions
        system_manager.set_service_set(svc_set, svc_arrival, self.max_arrival)
        system_manager.set_servers(edge, fog, cloud)

        system_manager.ranku = np.zeros(self.num_partitions)
        system_manager.calc_average()
        system_manager.calc_ranku(svc_set.partitions[0])

        return svc_set, system_manager


if __name__=="__main__":
    d = DAGDataSet(max_timeslot=24)
    d.data_gen()