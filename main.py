import numpy as np
from dag_data_generator import DAGDataSet
from genetic import Genetic, PSOGA, Greedy, HEFT
from operator import itemgetter
import time
import sys
import draw_result
import pickle



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
            # print("#timeslot {} x: {}".format(t, x))
            print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in dataset.system_manager.server.values()]))
            # print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in dataset.system_manager.server.values()]))
            # print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in dataset.system_manager.server.values()]))
            # print("#timeslot {} t: {}".format(t, dataset.system_manager.total_time_dp()))
            total_time.append(dataset.system_manager.total_time_dp())
            # print("finish_time", dataset.system_manager.finish_time)
            total_energy.append(sum([s.energy_consumption() for s in list(dataset.system_manager.request.values()) + list(dataset.system_manager.local.values()) + list(dataset.system_manager.edge.values())]))
            total_reward.append(dataset.system_manager.get_reward())
            dataset.system_manager.after_timeslot(deployed_server=x, timeslot=t)
        print("mean t: {:.5f}".format(np.sum(total_time, axis=None)), np.sum(np.array(total_time), axis=0) / dataset.num_timeslots)
        print("mean e: {:.5f}".format(sum(total_energy) / dataset.num_timeslots))
        print("mean r: {:.5f}".format(sum(total_reward) / dataset.num_timeslots))
        print("took: {:.5f} sec".format(took[idx]))
        result.append([np.sum(total_time, axis=None), sum(total_energy) / dataset.num_timeslots, sum(total_reward) / dataset.num_timeslots])
    print("\033[95m---------- " + algorithm_name + " Done! ----------\033[0m\n")
    return result



if __name__=="__main__":
    # for DAG recursive functions
    print('recursion limit', sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    test_num_services = 6
    test_num_servers = 10
    try:
        with open("outputs/net_manager_backup", "rb") as fp:
            net_manager = pickle.load(fp)
        test_dataset = DAGDataSet(num_timeslots=1, num_services=test_num_services, net_manager=net_manager, apply_partition=False)

    except:
        test_dataset = DAGDataSet(num_timeslots=1, num_services=test_num_services, apply_partition=False)
        with open("outputs/net_manager_backup", "wb") as fp:
            pickle.dump(test_dataset.system_manager.net_manager, fp)

    test_dataset.system_manager.set_env(deployed_server=np.full(shape=test_dataset.num_partitions, fill_value=list(test_dataset.system_manager.request.keys())[0], dtype=np.int32))
    print("Raspberry Pi computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in test_dataset.svc_set.services])
    print("Raspberry Pi computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in test_dataset.svc_set.services], "\n")
    test_dataset.system_manager.set_env(deployed_server=np.full(shape=test_dataset.num_partitions, fill_value=list(test_dataset.system_manager.local.keys())[0], dtype=np.int32))
    print("Jetson Nano computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in test_dataset.svc_set.services])
    print("Jetson Nano computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in test_dataset.svc_set.services], "\n")
    test_dataset.system_manager.set_env(deployed_server=np.full(shape=test_dataset.num_partitions, fill_value=list(test_dataset.system_manager.local.keys())[1], dtype=np.int32))
    print("Jetson TX2 computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in test_dataset.svc_set.services])
    print("Jetson TX2 computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in test_dataset.svc_set.services], "\n")
    test_dataset.system_manager.set_env(deployed_server=np.full(shape=test_dataset.num_partitions, fill_value=list(test_dataset.system_manager.edge.keys())[0], dtype=np.int32))
    print("Edge computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in test_dataset.svc_set.services])
    print("Edge computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in test_dataset.svc_set.services], "\n")

    start = time.time()
    x_lst = [np.array([[test_dataset.system_manager.net_manager.request_device[p.service.id] for p in test_dataset.svc_set.partitions] for t in range(test_dataset.num_timeslots)])]
    test_result = result(test_dataset, x_lst, took=[time.time()-start], algorithm_name="Test Local Only")

    start = time.time()
    x_lst = [np.full(shape=(test_dataset.num_timeslots, test_dataset.num_partitions), fill_value=list(test_dataset.system_manager.edge.keys())[0], dtype=np.int32)]
    test_result = result(test_dataset, x_lst, took=[time.time()-start], algorithm_name="Test Edge Only")

    heft = HEFT(dataset=test_dataset)
    heft.server_lst = list(test_dataset.system_manager.local.keys())[:test_num_servers] #+ list(test_dataset.system_manager.edge.keys())

    start = time.time()
    heft_x_lst = [heft.run_algo()]
    heft_result = result(test_dataset, heft_x_lst, took=[time.time()-start], algorithm_name="Test HEFT Algorithm (M={}, D={})".format(test_num_services, test_num_servers))

    print("\n========== Our Scheme Start ==========\n")

    result_by_services = []

    service_low = 3
    service_high = 9
    service_step = 3
    for num_services in range(service_low, service_high+1, service_step):
        print(":::::::::: M ==", num_services, "::::::::::\n")
    
        local_result = []
        edge_result = []
        greedy_result = []

        memetic_psoga_result = []
        memetic_genetic_result = []
        psoga_result = []
        genetic_result = []

        memetic_psoga_eval_lst = []
        memetic_genetic_eval_lst = []
        psoga_eval_lst = []
        genetic_eval_lst = []

        memetic_psoga_took_lst = []
        memetic_genetic_took_lst = []
        psoga_took_lst = []
        genetic_took_lst = []

        dataset = DAGDataSet(num_timeslots=1, num_services=num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition='horizontal')
        #dataset.system_manager.scheduling_policy = 'EFT'

        dataset.system_manager.set_env(deployed_server=np.full(shape=dataset.num_partitions, fill_value=list(dataset.system_manager.request.keys())[0], dtype=np.int32))
        print("Raspberry Pi computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in dataset.svc_set.services])
        print("Raspberry Pi computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in dataset.svc_set.services], "\n")
        dataset.system_manager.set_env(deployed_server=np.full(shape=dataset.num_partitions, fill_value=list(dataset.system_manager.local.keys())[0], dtype=np.int32))
        print("Jetson Nano computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in dataset.svc_set.services])
        print("Jetson Nano computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in dataset.svc_set.services], "\n")
        dataset.system_manager.set_env(deployed_server=np.full(shape=dataset.num_partitions, fill_value=list(dataset.system_manager.local.keys())[1], dtype=np.int32))
        print("Jetson TX2 computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in dataset.svc_set.services])
        print("Jetson TX2 computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in dataset.svc_set.services], "\n")
        dataset.system_manager.set_env(deployed_server=np.full(shape=dataset.num_partitions, fill_value=list(dataset.system_manager.edge.keys())[0], dtype=np.int32))
        print("Edge computation time", [sum([p.get_computation_time() for p in svc.partitions]) for svc in dataset.svc_set.services])
        print("Edge computing capacity", [sum([p.workload_size * p.deployed_server.computing_intensity[p.service.id] for p in svc.partitions]) for svc in dataset.svc_set.services], "\n")

        greedy = Greedy(dataset=dataset)
        psoga = PSOGA(dataset=dataset, num_particles=50, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        genetic = Genetic(dataset=dataset, num_solutions=50, mutation_ratio=0.1, cross_over_ratio=0.7)

        result_by_servers = []

        server_low = 0
        server_high = 10
        server_step = 2
        for num_servers in range(server_low, server_high+1, server_step):
            print(":::::::::: S ==", num_servers, "::::::::::\n")

            greedy.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())
            genetic.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())
            psoga.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())

            start = time.time()
            local_x_lst = [np.array([[dataset.system_manager.net_manager.request_device[p.service.id] for p in dataset.svc_set.partitions] for t in range(dataset.num_timeslots)])]
            local_took = [time.time() - start]
            local_result.append(result(dataset, local_x_lst, took=local_took, algorithm_name="Local Only"))

            start = time.time()
            edge_x_lst = [np.full(shape=(dataset.num_timeslots, dataset.num_partitions), fill_value=list(dataset.system_manager.edge.keys())[0], dtype=np.int32)]
            edge_took = [time.time() - start]
            edge_result.append(result(dataset, edge_x_lst, took=edge_took, algorithm_name="Edge Only"))

            start = time.time()
            greedy_x_lst = [greedy.run_algo()]
            greedy_took = [time.time() - start]
            greedy_result.append(result(dataset, greedy_x_lst, took=greedy_took, algorithm_name="Greedy Algorithm (M={}, D={})".format(num_services, num_servers)))

            # dataset.system_manager.init_env()
            # dataset.system_manager.set_env(deployed_server=greedy_x_lst[0][0])
            # deadline = dataset.system_manager.total_time_dp()
            # for svc in dataset.svc_set.services:
            #     svc.deadline = deadline[svc.id]

            iteration = 5
            loop = 300

            random_solution = None # psoga.generate_random_solutions()

            # temp = [genetic.run_algo(loop=loop, verbose=False, local_search=True) for _ in range(iteration)]
            # memetic_psoga_x_lst = [x for (x, e, t) in temp]
            # memetic_psoga_eval_lst.append([e for (x, e, t) in temp])
            # memetic_psoga_took = [t for (x, e, t) in temp]
            # memetic_psoga_took_lst.append(memetic_psoga_took)
            # memetic_psoga_result.append(sorted(result(dataset, memetic_psoga_x_lst, took=memetic_psoga_took, algorithm_name="Memetic-PSO-GA Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            temp = [genetic.run_algo(loop=loop, verbose=False, local_search=True) for _ in range(iteration)]
            memetic_genetic_x_lst = [x for (x, e, t) in temp]
            memetic_genetic_eval_lst.append([e for (x, e, t) in temp])
            memetic_genetic_took = [t for (x, e, t) in temp]
            memetic_genetic_took_lst.append(memetic_genetic_took)
            memetic_genetic_result.append(sorted(result(dataset, memetic_genetic_x_lst, took=memetic_genetic_took, algorithm_name="Memetic-Genetic Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            temp = [psoga.run_algo(loop=loop, verbose=False, local_search=False) for _ in range(iteration)]
            psoga_x_lst = [x for (x, e, t) in temp]
            psoga_eval_lst.append([e for (x, e, t) in temp])
            psoga_took = [t for (x, e, t) in temp]
            psoga_took_lst.append(psoga_took)
            psoga_result.append(sorted(result(dataset, psoga_x_lst, took=psoga_took, algorithm_name="PSO-GA Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            temp = [genetic.run_algo(loop=loop, verbose=False, local_search=False) for _ in range(iteration)]
            genetic_x_lst = [x for (x, e, t) in temp]
            genetic_eval_lst.append([e for (x, e, t) in temp])
            genetic_took = [t for (x, e, t) in temp]
            genetic_took_lst.append(genetic_took)
            genetic_result.append(sorted(result(dataset, genetic_x_lst, took=genetic_took, algorithm_name="Genetic Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

        result_by_services.append([local_result, edge_result, greedy_result, memetic_psoga_result, memetic_genetic_result, psoga_result, genetic_result, memetic_psoga_eval_lst, memetic_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, memetic_psoga_took_lst, memetic_genetic_took_lst, psoga_took_lst, genetic_took_lst, server_low, server_high, server_step, num_services])
        with open("outputs/results_backup_{}".format(num_services), "wb") as fp:
            pickle.dump(result_by_services, fp)


    replace = False
    replace_service = 0
    replace_server = 1
    if replace:
        with open("outputs/results_backup", "rb") as fp:
            temp = pickle.load(fp)
            temp[replace_service][replace_server] = result_by_services[0][0]
        with open("outputs/results_backup", "wb") as fp:
            pickle.dump(temp, fp)
    else:
        with open("outputs/results_backup", "wb") as fp:
            pickle.dump(result_by_services, fp)

    for result_by_servers in result_by_services:
        draw_result.by_num_service(*result_by_servers)