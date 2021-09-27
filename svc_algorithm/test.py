import algorithm_sim
import requester
from collections import Counter
import numpy as np
import sim_system


if __name__ == "__main__":
    r_sim = requester.RequestSim(1)
    r_sim.init(10, 2)
    r_sim.sim_start()
    r_sim.env.run(until=600)
    request = r_sim.result

    pop = Counter(map(lambda x: x[1], request))
    pop = [pop[i] for i in range(len(pop))]
    pop = np.array(pop)
    pop = pop / pop.sum()

    print(pop)

    env = sim_system.simpy.Environment()
    sim = sim_system.Simulator()
    num_edge = 10
    num_service = len(pop)
    edge_bandwidth = np.full(10, 100 * (1000 ** 2))  # 100Mbps
    cloud_bandwidth = 10 * (1000 ** 2)  # 10Mbps
    user = sim_system.User(edge_bandwidth, cloud_bandwidth)
    cloud_cp = 1000
    edge_cp = 100
    cloud = sim_system.Cloud(env, cloud_cp)
    cloud.set_bandwidth(cloud_bandwidth)
    edge_lst = [sim_system.Edge(env, 100, 8 * (1024 ** 3), edge_cp, idx) for idx in range(num_edge)]
    for edge in edge_lst:
        edge.set_bandwidth(np.full((num_edge, num_edge), edge_bandwidth))
    svc_lst = list()
    for service_idx in range(num_service):
        ms_lst = list()
        for layer_idx in range(np.random.randint(4, 11)):
            ms_lst.append(sim_system.MicroService(10, 50 * (1024 ** 2), 10 * 1024, 10, layer_idx))
        ms_set = sim_system.MicroServiceSet(10 * (1024 ** 2), np.random.random())
        ms_set.set_seq_micro_service(ms_lst)
        svc_lst.append(ms_set)
    svc_set = sim_system.ServiceSet(svc_lst)
    edge_cluster = sim_system.EdgeCluster(edge_lst, svc_set)
    sim.set_env(env, svc_set, user, cloud, edge_cluster)

    ga = algorithm_sim.Memetic(pop, edge_cluster, svc_set)
    k = ga.run_algo(100, 10**3)
    print(k)
    edge_cluster.set_k(k)
    # rl = algorithm.RL(sim)

    for time, item in request:
        sim.run_request_event(item, time)

    env.run(until=600)
