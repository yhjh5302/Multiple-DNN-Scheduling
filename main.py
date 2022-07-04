import numpy as np
from dag_data_generator import DAGDataSet
from genetic import Genetic, Layerwise_PSOGA, Greedy, HEFT, CPOP, PEFT
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
    print('recursion limit', sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    test_num_services = 1
    test_num_servers = 10
    
    try:
        with open("outputs/net_manager_backup", "rb") as fp:
            net_manager = pickle.load(fp)
        test_dataset = DAGDataSet(num_timeslots=1, num_services=test_num_services, net_manager=net_manager, apply_partition=False)

    except:
        print("except")
        test_dataset = DAGDataSet(num_timeslots=1, num_services=test_num_services, apply_partition=False)
        with open("outputs/net_manager_backup", "wb") as fp:
            pickle.dump(test_dataset.system_manager.net_manager, fp)

    print("\n========== Piecewise Scheme Start ==========\n")

    result_by_services = []

    service_low = 5
    service_high = 9
    service_step = 1
    for num_services in range(service_low, service_high+1, service_step):
        print(":::::::::: M ==", num_services, "::::::::::\n")
    
        local_result = []
        edge_result = []
        heft_u_result = []
        heft_d_result = []
        cpop_result = []
        peft_result = []
        greedy_u_result = []
        greedy_d_result = []

        greedy_psoga_result = []
        greedy_genetic_result = []
        psoga_result = []
        genetic_result = []

        greedy_psoga_eval_lst = []
        greedy_genetic_eval_lst = []
        psoga_eval_lst = []
        genetic_eval_lst = []

        greedy_psoga_took_lst = []
        greedy_genetic_took_lst = []
        psoga_took_lst = []
        genetic_took_lst = []

        dataset = DAGDataSet(num_timeslots=1, num_services=num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition='horizontal')
        layerwise_dataset = DAGDataSet(num_timeslots=1, num_services=num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition=False)

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

        heft = HEFT(dataset=dataset)
        cpop = CPOP(dataset=dataset)
        peft = PEFT(dataset=dataset)
        greedy = Greedy(dataset=dataset)
        psoga = Layerwise_PSOGA(dataset=layerwise_dataset, num_particles=50, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        genetic = Genetic(dataset=dataset, num_solutions=50, mutation_ratio=0.3, cross_over_ratio=0.7)

        result_by_servers = []

        server_low = 4
        server_high = 10
        server_step = 2
        for num_servers in range(server_low, server_high+1, server_step):
            print(":::::::::: S ==", num_servers, "::::::::::\n")

            heft.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())
            cpop.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())
            peft.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())
            greedy.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())
            genetic.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())
            psoga.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys())

            dataset.system_manager.scheduling_policy = 'noschedule'

            start = time.time()
            local_x_lst = [np.array([[dataset.system_manager.net_manager.request_device[p.service.id] for p in dataset.svc_set.partitions] for t in range(dataset.num_timeslots)])]
            local_took = [time.time() - start]
            local_result.append(result(dataset, local_x_lst, took=local_took, algorithm_name="Local Only"))

            start = time.time()
            edge_x_lst = [np.full(shape=(dataset.num_timeslots, dataset.num_partitions), fill_value=list(dataset.system_manager.edge.keys())[0], dtype=np.int32)]
            edge_took = [time.time() - start]
            edge_result.append(result(dataset, edge_x_lst, took=edge_took, algorithm_name="Edge Only"))

            heft.rank = 'rank_u'
            dataset.system_manager.scheduling_policy = 'rank_u'

            start = time.time()
            heft_x_lst = [heft.run_algo()]
            heft_took = [time.time() - start]
            heft_u_result.append(result(dataset, heft_x_lst, took=heft_took, algorithm_name="HEFT-U Algorithm (M={}, D={})".format(num_services, num_servers)))

            # heft.rank = 'rank_d'
            # dataset.system_manager.scheduling_policy = 'rank_d'

            # start = time.time()
            # heft_x_lst = [heft.run_algo()]
            # heft_took = [time.time() - start]
            # heft_d_result.append(result(dataset, heft_x_lst, took=heft_took, algorithm_name="HEFT-D Algorithm (M={}, D={})".format(num_services, num_servers)))

            dataset.system_manager.scheduling_policy = 'rank_u' # meaningless

            start = time.time()
            cpop_x_lst = [cpop.run_algo()]
            cpop_took = [time.time() - start]
            cpop_result.append(result(dataset, cpop_x_lst, took=cpop_took, algorithm_name="CPOP Algorithm (M={}, D={})".format(num_services, num_servers)))

            dataset.system_manager.scheduling_policy = 'rank_u' # meaningless

            start = time.time()
            peft_x_lst = [peft.run_algo()]
            peft_took = [time.time() - start]
            peft_result.append(result(dataset, peft_x_lst, took=peft_took, algorithm_name="PEFT Algorithm (M={}, D={})".format(num_services, num_servers)))

            # greedy.rank = 'rank_u'
            # dataset.system_manager.scheduling_policy = 'rank_u'

            # start = time.time()
            # greedy_x_lst = [greedy.run_algo_piecewise()]
            # greedy_took = [time.time() - start]
            # greedy_u_result.append(result(dataset, greedy_x_lst, took=greedy_took, algorithm_name="Greedy-U Algorithm (M={}, D={})".format(num_services, num_servers)))

            # greedy.rank = 'rank_d'
            # dataset.system_manager.scheduling_policy = 'rank_d'

            # start = time.time()
            # greedy_x_lst = [greedy.run_algo_piecewise()]
            # greedy_took = [time.time() - start]
            # greedy_d_result.append(result(dataset, greedy_x_lst, took=greedy_took, algorithm_name="Greedy-D Algorithm (M={}, D={})".format(num_services, num_servers)))

            # dataset.system_manager.init_env()
            # dataset.system_manager.set_env(deployed_server=greedy_x_lst[0][0])
            # deadline = dataset.system_manager.total_time_dp()
            # for svc in dataset.svc_set.services:
            #     svc.deadline = deadline[svc.id]

            iteration = 5

            # dataset.system_manager.calculate_rank_oct_schedule(list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()))
            dataset.system_manager.scheduling_policy = 'rank_u' # rank_u, earliest
            layerwise_dataset.system_manager.scheduling_policy = 'rank_u' # rank_u, earliest

            temp = [psoga.run_algo(loop=300, verbose=False, local_search=False) for _ in range(iteration)]
            psoga_x_lst = [x for (x, e, t) in temp]
            psoga_eval_lst.append([e for (x, e, t) in temp])
            psoga_took = [t for (x, e, t) in temp]
            psoga_took_lst.append(psoga_took)
            psoga_result.append(sorted(result(layerwise_dataset, psoga_x_lst, took=psoga_took, algorithm_name="Layerwise PSO-GA Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            temp = [genetic.run_algo(loop=50, verbose=False, local_search=True) for _ in range(iteration)]
            genetic_x_lst = [x for (x, e, t) in temp]
            genetic_eval_lst.append([e for (x, e, t) in temp])
            genetic_took = [t for (x, e, t) in temp]
            genetic_took_lst.append(genetic_took)
            genetic_result.append(sorted(result(dataset, genetic_x_lst, took=genetic_took, algorithm_name="Memetic Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

        del heft
        del greedy
        del genetic
        del psoga
        del dataset
        result_by_services.append([local_result, edge_result, heft_u_result, heft_d_result, cpop_result, peft_result, greedy_u_result, greedy_d_result, greedy_psoga_result, greedy_genetic_result, psoga_result, genetic_result, greedy_psoga_eval_lst, greedy_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, greedy_psoga_took_lst, greedy_genetic_took_lst, psoga_took_lst, genetic_took_lst, server_low, server_high, server_step, num_services])
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