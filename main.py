import numpy as np
from dag_data_generator import DAGDataSet
from algorithms.ServerOrderEvolutionary import ServerOrderGenetic, ServerOrderPSOGA
from algorithms.ServerEvolutionary import Genetic, PSOGA
from algorithms.Greedy import Local, Edge, HEFT, CPOP, PEFT, Greedy
from operator import itemgetter
import time
import sys
import draw_result
import pickle
import argparse



def result(dataset, action_lst, took, algorithm_name):
    print("\033[95m---------- " + algorithm_name + " Test! ----------\033[96m")
    result = []
    for idx, action in enumerate(action_lst):
        total_time = []
        total_energy = []
        total_reward = []
        dataset.system_manager.init_env()
        for t in range(dataset.num_timeslots):
            if isinstance(action, tuple):
                x = np.array(action[0][t])
                y = np.array(action[1][t])
                dataset.system_manager.set_env(deployed_server=x, execution_order=y)
            else:
                x = np.array(action[t])
                dataset.system_manager.set_env(deployed_server=x)
            # print("#timeslot {} x: {}".format(t, dataset.system_manager.deployed_server))
            # print("#timeslot {} y: {}".format(t, dataset.system_manager.execution_order))
            print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in dataset.system_manager.server.values()]))
            # print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in dataset.system_manager.server.values()]))
            # print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in dataset.system_manager.server.values()]))
            # print("#timeslot {} t: {}".format(t, dataset.system_manager.total_time_dp()))
            total_time.append(dataset.system_manager.total_time_dp())
            # print("finish_time", dataset.system_manager.finish_time)
            total_energy.append(sum([s.energy_consumption() for s in list(dataset.system_manager.request.values()) + list(dataset.system_manager.local.values()) + list(dataset.system_manager.edge.values())]))
            total_reward.append(dataset.system_manager.get_reward())
            dataset.system_manager.after_timeslot(deployed_server=x, timeslot=t)
        print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
        print("mean e: {:.5f}".format(sum(total_energy) / dataset.num_timeslots))
        print("mean r: {:.5f}".format(sum(total_reward) / dataset.num_timeslots))
        print("took: {:.5f} sec".format(took[idx]))
        result.append([np.max(total_time, axis=None), sum(total_energy) / dataset.num_timeslots, sum(total_reward) / dataset.num_timeslots])
    print("\033[95m---------- " + algorithm_name + " Done! ----------\033[0m\n")
    return result



if __name__=="__main__":
    # for DAG recursive functions
    print("recursion limit", sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser(description="_")
    parser.add_argument("--num_services", type=int, default=3, help="_")
    parser.add_argument("--num_servers", type=int, default=2, help="_")
    parser.add_argument("--bandwidth_ratio", type=float, default=1.0, help="_")
    parser.add_argument("--partitioning", type=str, choices=["Piecewise", "Layerwise"], default="Layerwise", help="_")
    parser.add_argument("--offloading", type=str, choices=["Local", "Edge", "HEFT", "CPOP", "PEFT", "Greedy", "PSOGA", "Genetic", "MemeticPSOGA", "MemeticGenetic"], default="Local", help="_")
    parser.add_argument("--iteration", type=int, default=1, help="_")
    args = parser.parse_args()

    test_num_services = 1
    test_num_servers = 10
    
    try: # network_parameter_loading
        with open("outputs/net_manager_backup", "rb") as fp:
            net_manager = pickle.load(fp)
        test_dataset = DAGDataSet(num_timeslots=1, num_services=test_num_services, net_manager=net_manager, apply_partition=False)
    except:
        print("except")
        test_dataset = DAGDataSet(num_timeslots=1, num_services=test_num_services, apply_partition=False)
        with open("outputs/net_manager_backup", "wb") as fp:
            pickle.dump(test_dataset.system_manager.net_manager, fp)

    print("\n========== {}-{} Scheme Start ==========\n".format(args.partitioning, args.offloading))

    print(":::::::::: N ==", args.num_services, "::::::::::\n")

    algorithm_result = []
    algorithm_took_lst = []
    algorithm_action_lst = []
    algorithm_eval_lst = []

    if args.partitioning == "Piecewise":
        dataset = DAGDataSet(num_timeslots=1, num_services=args.num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition="horizontal", graph_coarsening=True)
    elif args.partitioning == "Layerwise":
        dataset = DAGDataSet(num_timeslots=1, num_services=args.num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition=False, layer_coarsening=True)
    else:
        raise RuntimeError(args.offloading, "is not our partitioning method")
    dataset.system_manager.net_manager.bandwidth_change(args.bandwidth_ratio)

    dataset.system_manager.set_env(deployed_server=np.full(shape=dataset.num_partitions, fill_value=list(dataset.system_manager.request.keys())[0], dtype=np.int32))
    print("Raspberry Pi computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in dataset.svc_set.services])
    print("Raspberry Pi computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in dataset.svc_set.services], "\n")
    dataset.system_manager.set_env(deployed_server=np.full(shape=dataset.num_partitions, fill_value=list(dataset.system_manager.local.keys())[1], dtype=np.int32))
    print("Jetson TX2 computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in dataset.svc_set.services])
    print("Jetson TX2 computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in dataset.svc_set.services], "\n")
    dataset.system_manager.set_env(deployed_server=np.full(shape=dataset.num_partitions, fill_value=list(dataset.system_manager.edge.keys())[0], dtype=np.int32))
    print("Edge computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in dataset.svc_set.services])
    print("Edge computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in dataset.svc_set.services], "\n")

    if args.offloading == "Local":
        algorithm = Local(dataset=dataset)
        algorithm_parameter = { }
        dataset.system_manager.scheduling_policy = "noschedule"
    elif args.offloading == "Edge":
        algorithm = Edge(dataset=dataset)
        algorithm_parameter = { }
        dataset.system_manager.scheduling_policy = "noschedule"
    elif args.offloading == "HEFT":
        algorithm = HEFT(dataset=dataset)
        algorithm_parameter = { }
        algorithm.rank = "rank_u"
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.offloading == "CPOP":
        algorithm = CPOP(dataset=dataset)
        algorithm_parameter = { }
        algorithm.rank = "rank_u"
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.offloading == "PEFT":
        algorithm = PEFT(dataset=dataset)
        algorithm_parameter = { }
        algorithm.rank = "rank_oct"
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.offloading == "Greedy":
        algorithm = Greedy(dataset=dataset)
        algorithm_parameter = { }
        # algorithm.rank = "rank_oct"
        algorithm.rank = "rank_u"
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Piecewise" and args.offloading == "MemeticPSOGA":
        algorithm = PSOGA(dataset=dataset, num_particles=50, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        algorithm_parameter = { "loop": 300, "verbose": 100, "local_search": True, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Piecewise" and args.offloading == "MemeticGenetic":
        algorithm = Genetic(dataset=dataset, num_solutions=50, mutation_ratio=0.1, cross_over_ratio=0.9)
        algorithm_parameter = { "loop": 300, "verbose": 100, "local_search": True, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Piecewise" and args.offloading == "PSOGA":
        algorithm = PSOGA(dataset=dataset, num_particles=50, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        algorithm_parameter = { "loop": 600, "verbose": 100, "local_search": False, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Piecewise" and args.offloading == "Genetic":
        algorithm = Genetic(dataset=dataset, num_solutions=50, mutation_ratio=0.1, cross_over_ratio=0.9)
        algorithm_parameter = { "loop": 600, "verbose": 100, "local_search": False, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Layerwise" and args.offloading == "MemeticPSOGA":
        algorithm = ServerOrderPSOGA(dataset=dataset, num_particles=50, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        algorithm_parameter = { "loop": 300, "verbose": 100, "local_search": True, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Layerwise" and args.offloading == "MemeticGenetic":
        algorithm = ServerOrderGenetic(dataset=dataset, num_solutions=50, mutation_ratio=0.1, cross_over_ratio=0.9)
        algorithm_parameter = { "loop": 300, "verbose": 100, "local_search": True, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Layerwise" and args.offloading == "PSOGA":
        algorithm = ServerOrderPSOGA(dataset=dataset, num_particles=50, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        algorithm_parameter = { "loop": 600, "verbose": 100, "local_search": False, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    elif args.partitioning == "Layerwise" and args.offloading == "Genetic":
        algorithm = ServerOrderGenetic(dataset=dataset, num_solutions=50, mutation_ratio=0.1, cross_over_ratio=0.9)
        algorithm_parameter = { "loop": 600, "verbose": 100, "local_search": False, "early_exit_loop": 50 }
        dataset.system_manager.scheduling_policy = "rank_u"
    else:
        raise RuntimeError(args.offloading, "is not our algorithm")

    print(":::::::::: D ==", args.num_servers, "::::::::::\n")
    algorithm.server_lst = [0] + list(dataset.system_manager.local.keys())[:args.num_servers] + list(dataset.system_manager.edge.keys())

    temp = [algorithm.run_algo(**algorithm_parameter) for _ in range(args.iteration)]
    algorithm_x_lst = [x for (x, e, t) in temp]
    algorithm_action_lst.append(algorithm_x_lst)
    algorithm_eval_lst.append([e for (x, e, t) in temp])
    algorithm_took = [t for (x, e, t) in temp]
    algorithm_took_lst.append(algorithm_took)
    algorithm_result.append(sorted(result(dataset, algorithm_x_lst, took=algorithm_took, algorithm_name=args.partitioning+" "+args.offloading+" "+"Algorithm (N={}, D={})".format(args.num_services, args.num_servers)), key=itemgetter(2), reverse=True))

    with open("outputs/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format(args.partitioning, args.offloading, args.num_services, args.num_servers, args.bandwidth_ratio), "wb") as fp:
        pickle.dump([args.partitioning, args.offloading, algorithm_result, algorithm_took_lst, algorithm_eval_lst, algorithm_action_lst, args.num_services, args.num_servers], fp)