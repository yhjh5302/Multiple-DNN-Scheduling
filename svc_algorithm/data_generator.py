import numpy as np
from random import *
from svc_algorithm.server import *


def create_arrival_rate(num_services=10, minimum=12, maximum=26):
    return minimum + (maximum - minimum) * np.random.random(num_services)


class DataSet:
    def __init__(self):
        self.svc_set, self.system_manager = self.data_gen()
        self.num_containers = len(self.svc_set.container_set)
        self.num_services = len(self.svc_set.svc_set)
        self.num_edges = len(self.system_manager.edge)

    def data_gen(self, num_services=10, max_partitions=5, deadline_opt=(10, 100), num_edges=10, cpu_spec=(2, 4),
                 mem_spec=(64, 512), ipc=(10**12), fog_cpu=100, fog_mem=1024*1024*32, cloud_cpu=100, B_gw=25, B_fog=10, B_cl=5, P_dd_opt=(0.5,1)):
        # ipc -> TFLOPS
        # create system manager
        system_manager = SystemManager()

        # create service set
        svc_set = ServiceSet()
        for i in range(num_services):
            # create service
            deadline = uniform(deadline_opt[0], deadline_opt[1])
            svc = Service(deadline)

            # create partitions
            option = 0 #
            num_partitions = randint(1, max_partitions)
            svc.create_partitions(option, num_partitions)

            svc_set.add_services(svc)

        # create servers
        edge = list()
        for i in range(num_edges):
            cpu = randint(cpu_spec[0], cpu_spec[1])
            mem = 1024 * randint(mem_spec[0], mem_spec[1])
            edge.append(Edge(cpu, mem, ipc, system_manager=system_manager, id=i))
        fog = Server(fog_cpu, fog_mem, ipc, system_manager=system_manager)
        cloud = Server(cloud_cpu, float("inf"), ipc, system_manager=system_manager)

        # create network manager
        noise = 1
        channel_bandwidth = 25
        channel_gain = 1
        net_manager = NetworkManager(noise, channel_bandwidth, channel_gain)
        net_manager.B_gw = B_gw
        net_manager.B_fog = B_fog
        net_manager.B_cl = B_cl
        net_manager.P_dd = np.zeros(shape=(num_edges, num_edges))
        for i in range(num_edges):
            for j in range(i + 1, num_edges):
                net_manager.P_dd[i, j] = net_manager.P_dd[j, i] = uniform(P_dd_opt[0], P_dd_opt[1])
            net_manager.P_dd[i, i] = 0
        net_manager.cal_b_dd()

        # init system manager
        system_manager.set_service_set(svc_set)
        system_manager.set_servers(edge, fog, cloud)
        system_manager.net_manager = net_manager

        system_manager.init_servers()
        system_manager.calculate_high_layer()
        system_manager.set_arrival_rate(create_arrival_rate(num_services))
        system_manager.update_edge_computing_time()
        system_manager.update_transmission_mats()

        # sequential partition deployment algorithm
        y = spd_algorithm(system_manager, 0.3333, 0.3333, 0.3333)
        system_manager.set_y_mat(y)

        return svc_set, system_manager


if __name__=="__main__":
    d = DataSet()
    d.data_gen()