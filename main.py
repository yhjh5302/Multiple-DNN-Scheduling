import numpy as np
from dag_data_generator import DAGDataSet
from genetic import Genetic, PSOGA, Layerwise_PSOGA, HEFT, CPOP, PEFT, Greedy
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
    print('recursion limit', sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    parser = argparse.ArgumentParser(description="_")
    parser.add_argument('--service_low', type=int, default=3, help="_")
    parser.add_argument('--service_high', type=int, default=9, help="_")
    parser.add_argument('--service_step', type=int, default=1, help="_")
    parser.add_argument('--server_low', type=int, default=0, help="_")
    parser.add_argument('--server_high', type=int, default=10, help="_")
    parser.add_argument('--server_step', type=int, default=2, help="_")
    parser.add_argument('--bandwidth_ratio', type=float, default=1.0, help="_")
    args = parser.parse_args()

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

    service_low = args.service_low
    service_high = args.service_high
    service_step = args.service_step
    for num_services in range(service_low, service_high+1, service_step):
        print(":::::::::: M ==", num_services, "::::::::::\n")

        local_result = []
        edge_result = []
        heft_result = []
        cpop_result = []
        peft_result = []
        greedy_result = []
        layerwise_heft_result = []
        layerwise_cpop_result = []
        layerwise_peft_result = []
        layerwise_psoga_result = []
        layerwise_genetic_result = []
        psoga_result = []
        genetic_result = []

        layerwise_psoga_eval_lst = []
        layerwise_genetic_eval_lst = []
        psoga_eval_lst = []
        genetic_eval_lst = []

        local_took_lst = []
        edge_took_lst = []
        heft_took_lst = []
        cpop_took_lst = []
        peft_took_lst = []
        greedy_took_lst = []
        layerwise_heft_took_lst = []
        layerwise_cpop_took_lst = []
        layerwise_peft_took_lst = []
        layerwise_psoga_took_lst = []
        layerwise_genetic_took_lst = []
        psoga_took_lst = []
        genetic_took_lst = []

        local_action_lst = []
        edge_action_lst = []
        heft_action_lst = []
        cpop_action_lst = []
        peft_action_lst = []
        greedy_action_lst = []
        layerwise_heft_action_lst = []
        layerwise_cpop_action_lst = []
        layerwise_peft_action_lst = []
        layerwise_psoga_action_lst = []
        layerwise_genetic_action_lst = []
        psoga_action_lst = []
        genetic_action_lst = []

        dataset = DAGDataSet(num_timeslots=1, num_services=num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition='horizontal')
        layerwise_dataset = DAGDataSet(num_timeslots=1, num_services=num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition=False, layer_coarsening=False)
        layerwise_coarsened_dataset = DAGDataSet(num_timeslots=1, num_services=num_services, net_manager=test_dataset.system_manager.net_manager, apply_partition=False, layer_coarsening=True)
        dataset.system_manager.net_manager.bandwidth_change(args.bandwidth_ratio)
        layerwise_dataset.system_manager.net_manager.bandwidth_change(args.bandwidth_ratio)

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
        layerwise_heft = HEFT(dataset=layerwise_dataset)
        layerwise_cpop = CPOP(dataset=layerwise_dataset)
        layerwise_peft = PEFT(dataset=layerwise_dataset)
        layerwise_psoga = Layerwise_PSOGA(dataset=layerwise_coarsened_dataset, num_particles=100, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        layerwise_genetic = Genetic(dataset=layerwise_coarsened_dataset, num_solutions=100, mutation_ratio=0.3, cross_over_ratio=0.7)
        psoga = PSOGA(dataset=dataset, num_particles=50, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9)
        genetic = Genetic(dataset=dataset, num_solutions=50, mutation_ratio=0.3, cross_over_ratio=0.7)

        result_by_servers = []

        server_low = args.server_low
        server_high = args.server_high
        server_step = args.server_step
        for num_servers in range(server_low, server_high+1, server_step):
            print(":::::::::: S ==", num_servers, "::::::::::\n")

            layerwise_heft.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            layerwise_cpop.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            layerwise_peft.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            heft.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            cpop.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            peft.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            greedy.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            layerwise_genetic.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            layerwise_psoga.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            genetic.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]
            psoga.server_lst = list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()) # + [0]

            dataset.system_manager.scheduling_policy = 'noschedule'
            layerwise_dataset.system_manager.scheduling_policy = 'noschedule'

            start = time.time()
            local_x_lst = [np.array([[layerwise_dataset.system_manager.net_manager.request_device[p.service.id] for p in layerwise_dataset.svc_set.partitions] for t in range(layerwise_dataset.num_timeslots)])]
            local_took = [time.time() - start]
            local_result.append(result(layerwise_dataset, local_x_lst, took=local_took, algorithm_name="Local Only"))

            start = time.time()
            edge_x_lst = [np.full(shape=(layerwise_dataset.num_timeslots, layerwise_dataset.num_partitions), fill_value=list(layerwise_dataset.system_manager.edge.keys())[0], dtype=np.int32)]
            edge_took = [time.time() - start]
            edge_result.append(result(layerwise_dataset, edge_x_lst, took=edge_took, algorithm_name="Edge Only"))

            heft.rank = 'rank_u'
            dataset.system_manager.scheduling_policy = 'rank_u'

            start = time.time()
            heft_x_lst = [heft.run_algo()]
            heft_action_lst.append(heft_x_lst)
            heft_took = [time.time() - start]
            heft_took_lst.append(heft_took)
            heft_result.append(result(dataset, heft_x_lst, took=heft_took, algorithm_name="Hybrid HEFT Algorithm (M={}, D={})".format(num_services, num_servers)))

            # dataset.system_manager.scheduling_policy = 'rank_u'

            # start = time.time()
            # cpop_x_lst = [cpop.run_algo()]
            # cpop_action_lst.append(cpop_x_lst)
            # cpop_took = [time.time() - start]
            # cpop_took_lst.append(cpop_took)
            # cpop_result.append(result(dataset, cpop_x_lst, took=cpop_took, algorithm_name="Hybrid CPOP Algorithm (M={}, D={})".format(num_services, num_servers)))

            # dataset.system_manager.scheduling_policy = 'rank_u'

            # start = time.time()
            # peft_x_lst = [peft.run_algo()]
            # peft_action_lst.append(peft_x_lst)
            # peft_took = [time.time() - start]
            # peft_took_lst.append(peft_took)
            # peft_result.append(result(dataset, peft_x_lst, took=peft_took, algorithm_name="Hybrid PEFT Algorithm (M={}, D={})".format(num_services, num_servers)))

            # greedy.rank = 'rank_oct'
            # dataset.system_manager.calculate_rank_oct_schedule(list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()))
            greedy.rank = 'rank_u'
            dataset.system_manager.scheduling_policy = 'rank_u' # meaningless

            start = time.time()
            greedy_x_lst = [greedy.run_algo()]
            greedy_action_lst.append(greedy_x_lst)
            greedy_took = [time.time() - start]
            greedy_took_lst.append(greedy_took)
            greedy_result.append(result(dataset, greedy_x_lst, took=greedy_took, algorithm_name="Greedy Algorithm (M={}, D={})".format(num_services, num_servers)))

            layerwise_heft.rank = 'rank_u'
            layerwise_dataset.system_manager.scheduling_policy = 'rank_u'

            start = time.time()
            layerwise_heft_x_lst = [layerwise_heft.run_algo()]
            layerwise_heft_action_lst.append(layerwise_heft_x_lst)
            layerwise_heft_took = [time.time() - start]
            layerwise_heft_took_lst.append(layerwise_heft_took)
            layerwise_heft_result.append(result(layerwise_dataset, layerwise_heft_x_lst, took=layerwise_heft_took, algorithm_name="Vertical HEFT Algorithm (M={}, D={})".format(num_services, num_servers)))

            # layerwise_dataset.system_manager.scheduling_policy = 'rank_u' # meaningless

            # start = time.time()
            # layerwise_cpop_x_lst = [layerwise_cpop.run_algo()]
            # layerwise_cpop_action_lst.append(layerwise_cpop_x_lst)
            # layerwise_cpop_took = [time.time() - start]
            # layerwise_cpop_took_lst.append(layerwise_cpop_took)
            # layerwise_cpop_result.append(result(layerwise_dataset, layerwise_cpop_x_lst, took=layerwise_cpop_took, algorithm_name="Vertical CPOP Algorithm (M={}, D={})".format(num_services, num_servers)))

            # layerwise_dataset.system_manager.scheduling_policy = 'rank_u' # meaningless

            # start = time.time()
            # layerwise_peft_x_lst = [layerwise_peft.run_algo()]
            # layerwise_peft_action_lst.append(layerwise_peft_x_lst)
            # layerwise_peft_took = [time.time() - start]
            # layerwise_peft_took_lst.append(layerwise_peft_took)
            # layerwise_peft_result.append(result(layerwise_dataset, layerwise_peft_x_lst, took=layerwise_peft_took, algorithm_name="Vertical PEFT Algorithm (M={}, D={})".format(num_services, num_servers)))

            iteration = 5

            # layerwise_coarsened_dataset.system_manager.calculate_rank_oct_schedule(list(dataset.system_manager.local.keys())[:num_servers] + list(dataset.system_manager.edge.keys()))
            layerwise_coarsened_dataset.system_manager.scheduling_policy = 'rank_u' # rank_u, earliest

            temp = [layerwise_psoga.run_algo(loop=1000, verbose=False, local_search=False, early_exit_loop=100) for _ in range(iteration)] # greedy_solution=[np.concatenate(layerwise_heft_x_lst[0], axis=1), np.concatenate(layerwise_cpop_x_lst[0], axis=1), np.concatenate(layerwise_peft_x_lst[0], axis=1)]
            layerwise_psoga_x_lst = [x for (x, e, t) in temp]
            layerwise_psoga_action_lst.append(layerwise_psoga_x_lst)
            layerwise_psoga_eval_lst.append([e for (x, e, t) in temp])
            layerwise_psoga_took = [t for (x, e, t) in temp]
            layerwise_psoga_took_lst.append(layerwise_psoga_took)
            layerwise_psoga_result.append(sorted(result(layerwise_coarsened_dataset, layerwise_psoga_x_lst, took=layerwise_psoga_took, algorithm_name="Vertical PSO-GA Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            # temp = [layerwise_psoga.run_algo(loop=1000, verbose=False, local_search=True, early_exit_loop=50) for _ in range(iteration)] # greedy_solution=[np.concatenate(layerwise_heft_x_lst[0], axis=1), np.concatenate(layerwise_cpop_x_lst[0], axis=1), np.concatenate(layerwise_peft_x_lst[0], axis=1)]
            # psoga_x_lst = [x for (x, e, t) in temp]
            # psoga_action_lst.append(psoga_x_lst)
            # psoga_eval_lst.append([e for (x, e, t) in temp])
            # psoga_took = [t for (x, e, t) in temp]
            # psoga_took_lst.append(psoga_took)
            # psoga_result.append(sorted(result(layerwise_coarsened_dataset, psoga_x_lst, took=psoga_took, algorithm_name="Hybrid PSO-GA Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            # temp = [layerwise_genetic.run_algo(loop=50, verbose=False, local_search=True, greedy_solution=layerwise_heft_x_lst[0][0], refinement=False, early_exit_loop=20) for _ in range(iteration)]
            # layerwise_genetic_x_lst = [x for (x, e, t) in temp]
            # layerwise_genetic_action_lst.append(layerwise_genetic_x_lst)
            # layerwise_genetic_eval_lst.append([e for (x, e, t) in temp])
            # layerwise_genetic_took = [t for (x, e, t) in temp]
            # layerwise_genetic_took_lst.append(layerwise_genetic_took)
            # layerwise_genetic_result.append(sorted(result(layerwise_coarsened_dataset, layerwise_genetic_x_lst, took=layerwise_genetic_took, algorithm_name="Layerwise Memetic Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            # temp = [psoga.run_algo(loop=300, verbose=1, local_search=True, refinement=True, early_exit_loop=30) for _ in range(iteration)]
            # psoga_x_lst = [x for (x, e, t) in temp]
            # psoga_action_lst.append(psoga_x_lst)
            # psoga_eval_lst.append([e for (x, e, t) in temp])
            # psoga_took = [t for (x, e, t) in temp]
            # psoga_took_lst.append(psoga_took)
            # psoga_result.append(sorted(result(dataset, psoga_x_lst, took=psoga_took, algorithm_name="Hybrid PSO-GA Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))

            # temp = [genetic.run_algo(loop=300, verbose=False, local_search=False, refinement=True, early_exit_loop=30) for _ in range(iteration)]
            # genetic_x_lst = [x for (x, e, t) in temp]
            # genetic_action_lst.append(genetic_x_lst)
            # genetic_eval_lst.append([e for (x, e, t) in temp])
            # genetic_took = [t for (x, e, t) in temp]
            # genetic_took_lst.append(genetic_took)
            # genetic_result.append(sorted(result(dataset, genetic_x_lst, took=genetic_took, algorithm_name="Hybrid Genetic Algorithm (M={}, D={})".format(num_services, num_servers)), key=itemgetter(2), reverse=True))
            
            with open("outputs/results_backup_service_{}_server_{}_bandwidth_{}".format(num_services, num_servers, args.bandwidth_ratio), "wb") as fp:
                pickle.dump([local_result, edge_result, heft_result, cpop_result, peft_result, greedy_result, layerwise_heft_result, layerwise_cpop_result, layerwise_peft_result, layerwise_psoga_result, layerwise_genetic_result, psoga_result, genetic_result, layerwise_psoga_eval_lst, layerwise_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, local_took_lst, edge_took_lst, heft_took_lst, cpop_took_lst, peft_took_lst, greedy_took_lst, layerwise_heft_took_lst, layerwise_cpop_took_lst, layerwise_peft_took_lst, layerwise_psoga_took_lst, layerwise_genetic_took_lst, psoga_took_lst, genetic_took_lst, local_action_lst, edge_action_lst, heft_action_lst, cpop_action_lst, peft_action_lst, greedy_action_lst, layerwise_heft_action_lst, layerwise_cpop_action_lst, layerwise_peft_action_lst, layerwise_psoga_action_lst, layerwise_genetic_action_lst, psoga_action_lst, genetic_action_lst, server_low, server_high, server_step, num_services], fp)

        del heft
        del cpop
        del peft
        del genetic
        del psoga
        del dataset