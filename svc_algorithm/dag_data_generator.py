import numpy as np
import random
from svc_algorithm.dag_server import *


class DAGDataSet:
    def __init__(self, max_timeslot):
        self.max_timeslot = max_timeslot
        self.svc_set, self.system_manager = self.data_gen()
        self.num_containers = len(self.svc_set.container_set)
        self.num_services = len(self.svc_set.svc_set)

    def create_arrival_rate(self, num_services, minimum, maximum):
        return minimum + (maximum - minimum) * np.random.random(num_services)

    def data_gen(self, num_services=5, max_partitions=1, deadline_opt=(10, 20), num_edges=7, num_fogs=2, num_clouds=1,
                ipc=(10**12), B_gw=1024*25, B_fog=1024*10, B_cl=1024*5, P_dd_opt=(0.5,1)):
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
            num_partitions = max_partitions
            svc.create_partitions(opt=(0, ((1, num_partitions), (0, num_partitions), (1, ((0, num_partitions), (0, num_partitions))), (0, 1))))

            svc_set.add_services(svc)

        # create arrival rate table
        self.max_arrival = 50
        self.min_arrival = 30

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
                cpu = random.randint(400, 2000) / 1000 # Tflops
                mem = random.randint(2, 4) * 1024 * 1024 # KB
            elif device == 'mobile':
                cpu = random.randint(1000, 2000) / 1000 # Tflops
                mem = random.randint(4, 16) * 1024 * 1024 # KB
            else:
                raise RuntimeError('Unknown device type {}'.format(device))
            edge[i] = Server(cpu, mem, ipc, system_manager=system_manager, id=i)
        for i in range(num_edges, num_edges + num_fogs):
            fog_cpu = random.randint(5, 10) # Tflops
            fog_mem = random.randint(16, 32) * 1024 * 1024 # KB
            fog[i] = Server(fog_cpu, fog_mem, ipc, system_manager=system_manager, id=i)
        for i in range(num_edges + num_fogs, num_edges + num_fogs + num_clouds):
            cloud_cpu = random.randint(100, 100) # Tflops
            cloud_mem = random.randint(1024, 1024) * 1024 * 1024 # KB
            cloud[i] = Server(cloud_cpu, cloud_mem, ipc, system_manager=system_manager, id=i)

        # create network manager
        noise = 1
        channel_bandwidth = 1024*25
        channel_gain = 1
        net_manager = NetworkManager(channel_bandwidth, channel_gain, noise)
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
        system_manager.set_service_set(svc_set, svc_arrival)
        system_manager.set_servers(edge, fog, cloud)
        system_manager.net_manager = net_manager
        system_manager.num_servers = self.num_servers
        system_manager.num_containers = len(svc_set.container_set)

        min_x = np.zeros_like(svc_set.container_set)
        for container in svc_set.container_set:
            min_x[container.id] = container.computation_amount * 5
        system_manager.init_servers(min_x)
        return svc_set, system_manager


if __name__=="__main__":
    d = DAGDataSet()
    d.data_gen()