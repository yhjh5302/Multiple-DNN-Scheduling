import os
import numpy as np
from matplotlib import pyplot as plt

# Get results by the number of services
def by_num_service(local_result, edge_result, heft_result, cpop_result, peft_result, greedy_result, layerwise_heft_result, layerwise_cpop_result, layerwise_peft_result, layerwise_psoga_result, layerwise_genetic_result, psoga_result, genetic_result, layerwise_psoga_eval_lst, layerwise_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, local_took_lst, edge_took_lst, heft_took_lst, cpop_took_lst, peft_took_lst, greedy_took_lst, layerwise_heft_took_lst, layerwise_cpop_took_lst, layerwise_peft_took_lst, layerwise_psoga_took_lst, layerwise_genetic_took_lst, psoga_took_lst, genetic_took_lst, local_action_lst, edge_action_lst, heft_action_lst, cpop_action_lst, peft_action_lst, greedy_action_lst, layerwise_heft_action_lst, layerwise_cpop_action_lst, layerwise_peft_action_lst, layerwise_psoga_action_lst, layerwise_genetic_action_lst, psoga_action_lst, genetic_action_lst, server_low, server_high, server_step, num_services, bandwidth):
    # [0,1,2]: 0 - num_devices, 1 - num_loop, 2 - [time,energy,reward]
    
    # algorithms = ['User Device', 'Edge', 'Vertical HEFT', 'Vertical CPOP', 'Vertical PEFT', 'Vertical PSO-GA', 'Vertical Genetic', 'Hybrid CPOP', 'Hybrid PEFT', 'Hybrid HEFT', 'Hybrid PSO-GA', 'Hybrid Genetic']
    algorithms = ['Vertical HEFT', 'Hybrid HEFT', 'Hybrid Heuristic']
    graph_type = 'line'

    # hatch = {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    if graph_type == 'bar':
        styles = [
            {'color':'white', 'edgecolor':'black', 'hatch':'///' },
            {'color':'deepskyblue', 'edgecolor':'black', 'hatch':'\\\\\\' },
            {'color':'yellow', 'edgecolor':'black', 'hatch':'|||' },
            {'color':'tomato', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'lime', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'orange', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'purple', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'red', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'blue', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'green', 'edgecolor':'black', 'hatch':'xxx' },
        ]
        
    elif graph_type == 'line':
        styles = [
            # {'c':'red', 'linestyle':'--', 'marker':'x', 'markersize':6},
            # {'c':'blue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            # {'c':'green', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'deeppink', 'linestyle':'--', 'marker':'x', 'markersize':6},
            {'c':'dodgerblue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            {'c':'limegreen', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'orange', 'linestyle':'--', 'marker':'s', 'markersize':6},
        ]

    x_range = range(server_low, server_high+1, server_step)
    x = [x for x in x_range]
    y = []
    plt.figure(figsize=(8,6))
    plt.title("Number of DNN = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services <= 3:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services <= 6:
        plt.yticks(np.arange(0, 1000+1, 50))
    else:
        plt.yticks(np.arange(0, 1000+1, 100))
    plt.ylim(0, np.max(np.array(heft_result)[:,:,0], axis=None) * 1200)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,0] * 1000
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,0] * 1000
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,0] * 1000
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,0] * 1000
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,0] * 1000
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,0] * 1000
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,0] * 1000
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "latency", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            print(x, avg)
            plt.plot(x, avg, label=algorithm, **styles[idx])
            # for index in range(len(x)):
            #     plt.text(x[index], np.round(avg, 2)[index], np.round(avg, 2)[index], size=8)
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Total completion time (ms)')
    plt.savefig(os.path.join(dir, 'outputs', 'delay_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    plt.title("Number of DNN = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 200+1, 1))
    elif num_services < 5:
        plt.yticks(np.arange(0, 200+1, 2))
    else:
        plt.yticks(np.arange(0, 200+1, 10))
    plt.ylim(0, np.max(np.array(heft_result)[:,:,1], axis=None) * 1.1)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,1]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,1]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,1]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,1]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,1]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,1]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,1]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,1]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,1]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,1]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,1]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,1]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,1]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "energy", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('IoT Energy Consumption (mJ)')
    plt.savefig(os.path.join(dir, 'outputs', 'energy_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    plt.title("Number of DNN = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(-1, 1+1, 0.01))
    elif num_services < 4:
        plt.yticks(np.arange(-1, 1+1, 0.02))
    else:
        plt.yticks(np.arange(-1, 1+1, 0.05))
    plt.ylim(np.min(np.array(heft_result)[:,:,2], axis=None) * 1.1, 0)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,2]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,2]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,2]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,2]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,2]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,2]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,2]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,2]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,2]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,2]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,2]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,2]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "reward", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(dir, 'outputs', 'reward_devices_M={}.png'.format(num_services)))
    plt.clf()

    algorithms = ['Vertical HEFT', 'Hybrid HEFT', 'Hybrid Heuristic']
    y = []
    plt.title("Number of DNN = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 10000+1, 10))
    elif num_services < 4:
        plt.yticks(np.arange(0, 10000+1, 10))
    elif num_services < 5:
        plt.yticks(np.arange(0, 10000+1, 10))
    elif num_services < 7:
        plt.yticks(np.arange(0, 10000+1, 10))
    elif num_services < 8:
        plt.yticks(np.arange(0, 10000+1, 10))
    else:
        plt.yticks(np.arange(0, 10000+1, 10))
    plt.ylim(0, max(np.mean(greedy_took_lst, axis=1)) * 1.1)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_took_lst)[:,:]
        elif algorithm == 'Edge':
            avg = np.array(edge_took_lst)[:,:]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_took_lst)[:,:]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_took_lst)[:,:]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_took_lst)[:,:]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_took_lst)[:,:]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_took_lst)[:,:]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_took_lst)[:,:]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_took_lst)[:,:]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_took_lst)[:,:]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_took_lst)[:,:]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_took_lst)[:,:]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_took_lst)[:,:]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "time", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Execution Time (sec)')
    plt.savefig(os.path.join(dir, 'outputs', 'execution_time_devices_M={}.png'.format(num_services)))
    plt.clf()

# Get results by the number of services
def by_num_servers(local_result, edge_result, heft_result, cpop_result, peft_result, greedy_result, layerwise_heft_result, layerwise_cpop_result, layerwise_peft_result, layerwise_psoga_result, layerwise_genetic_result, psoga_result, genetic_result, layerwise_psoga_eval_lst, layerwise_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, local_took_lst, edge_took_lst, heft_took_lst, cpop_took_lst, peft_took_lst, greedy_took_lst, layerwise_heft_took_lst, layerwise_cpop_took_lst, layerwise_peft_took_lst, layerwise_psoga_took_lst, layerwise_genetic_took_lst, psoga_took_lst, genetic_took_lst, local_action_lst, edge_action_lst, heft_action_lst, cpop_action_lst, peft_action_lst, greedy_action_lst, layerwise_heft_action_lst, layerwise_cpop_action_lst, layerwise_peft_action_lst, layerwise_psoga_action_lst, layerwise_genetic_action_lst, psoga_action_lst, genetic_action_lst, server_low, server_high, server_step, num_services, bandwidth):
    # [0,1,2]: 0 - num_devices, 1 - num_loop, 2 - [time,energy,reward]
    
    # algorithms = ['User Device', 'Edge', 'Vertical HEFT', 'Vertical CPOP', 'Vertical PEFT', 'Vertical PSO-GA', 'Vertical Genetic', 'Hybrid CPOP', 'Hybrid PEFT', 'Hybrid HEFT', 'Hybrid PSO-GA', 'Hybrid Genetic']
    algorithms = ['Vertical HEFT', 'Hybrid HEFT', 'Hybrid Heuristic']
    graph_type = 'line'

    # hatch = {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    if graph_type == 'bar':
        styles = [
            {'color':'white', 'edgecolor':'black', 'hatch':'///' },
            {'color':'deepskyblue', 'edgecolor':'black', 'hatch':'\\\\\\' },
            {'color':'yellow', 'edgecolor':'black', 'hatch':'|||' },
            {'color':'tomato', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'lime', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'orange', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'purple', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'red', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'blue', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'green', 'edgecolor':'black', 'hatch':'xxx' },
        ]
        
    elif graph_type == 'line':
        styles = [
            # {'c':'red', 'linestyle':'--', 'marker':'x', 'markersize':6},
            # {'c':'blue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            # {'c':'green', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'deeppink', 'linestyle':'--', 'marker':'x', 'markersize':6},
            {'c':'dodgerblue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            {'c':'limegreen', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'orange', 'linestyle':'--', 'marker':'s', 'markersize':6},
        ]

    x_range = range(server_low, server_high+1, server_step)
    x = [x for x in x_range]
    y = []
    plt.figure(figsize=(8,5))
    plt.title("Number of IoT Devices = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services <= 3:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services <= 6:
        plt.yticks(np.arange(0, 1000+1, 50))
    else:
        plt.yticks(np.arange(0, 1000+1, 100))
    plt.ylim(0, np.max(np.array(heft_result)[:,:,0], axis=None) * 1200)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,0] * 1000
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,0] * 1000
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,0] * 1000
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,0] * 1000
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,0] * 1000
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,0] * 1000
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,0] * 1000
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "latency", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
            # for index in range(len(x)):
            #     plt.text(x[index], np.round(avg, 2)[index], np.round(avg, 2)[index], size=8)
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of Services')
    plt.ylabel('Total completion time (ms)')
    plt.savefig(os.path.join(dir, 'outputs', 'delay_services_D={}.png'.format(num_services)))
    plt.clf()

    y = []
    plt.title("Number of IoT Devices = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 200+1, 1))
    elif num_services < 5:
        plt.yticks(np.arange(0, 200+1, 2))
    else:
        plt.yticks(np.arange(0, 200+1, 10))
    plt.ylim(0, np.max(np.array(heft_result)[:,:,1], axis=None) * 1.1)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,1]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,1]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,1]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,1]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,1]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,1]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,1]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,1]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,1]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,1]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,1]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,1]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,1]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "energy", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of Services')
    plt.ylabel('IoT Energy Consumption (mJ)')
    plt.savefig(os.path.join(dir, 'outputs', 'energy_services_D={}.png'.format(num_services)))
    plt.clf()

    y = []
    plt.title("Number of IoT Devices = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(-1, 1+1, 0.01))
    elif num_services < 4:
        plt.yticks(np.arange(-1, 1+1, 0.02))
    else:
        plt.yticks(np.arange(-1, 1+1, 0.05))
    plt.ylim(np.min(np.array(heft_result)[:,:,2], axis=None) * 1.1, 0)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,2]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,2]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,2]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,2]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,2]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,2]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,2]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,2]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,2]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,2]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,2]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,2]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "reward", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of Services')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(dir, 'outputs', 'reward_services_D={}.png'.format(num_services)))
    plt.clf()

    algorithms = ['Vertical HEFT', 'Hybrid HEFT', 'Hybrid Heuristic']
    y = []
    plt.title("Number of IoT Devices = {}, CCR = {}".format(num_services, bandwidth))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 1000+1, 1))
    elif num_services < 4:
        plt.yticks(np.arange(0, 1000+1, 5))
    elif num_services < 5:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services < 7:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services < 8:
        plt.yticks(np.arange(0, 1000+1, 20))
    else:
        plt.yticks(np.arange(0, 1000+1, 20))
    plt.ylim(0, max(np.mean(greedy_took_lst, axis=1)) * 1.2)
    plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_took_lst)[:,:]
        elif algorithm == 'Edge':
            avg = np.array(edge_took_lst)[:,:]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_took_lst)[:,:]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_took_lst)[:,:]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_took_lst)[:,:]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_took_lst)[:,:]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_took_lst)[:,:]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_took_lst)[:,:]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_took_lst)[:,:]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_took_lst)[:,:]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_took_lst)[:,:]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_took_lst)[:,:]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_took_lst)[:,:]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "time", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of Services')
    plt.ylabel('Execution Time (sec)')
    plt.savefig(os.path.join(dir, 'outputs', 'execution_time_services_D={}.png'.format(num_services)))
    plt.clf()

# Get results by the number of services
def by_num_bandwidths(local_result, edge_result, heft_result, cpop_result, peft_result, greedy_result, layerwise_heft_result, layerwise_cpop_result, layerwise_peft_result, layerwise_psoga_result, layerwise_genetic_result, psoga_result, genetic_result, layerwise_psoga_eval_lst, layerwise_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, local_took_lst, edge_took_lst, heft_took_lst, cpop_took_lst, peft_took_lst, greedy_took_lst, layerwise_heft_took_lst, layerwise_cpop_took_lst, layerwise_peft_took_lst, layerwise_psoga_took_lst, layerwise_genetic_took_lst, psoga_took_lst, genetic_took_lst, local_action_lst, edge_action_lst, heft_action_lst, cpop_action_lst, peft_action_lst, greedy_action_lst, layerwise_heft_action_lst, layerwise_cpop_action_lst, layerwise_peft_action_lst, layerwise_psoga_action_lst, layerwise_genetic_action_lst, psoga_action_lst, genetic_action_lst, num_services, num_servers, bandwidth):
    # [0,1,2]: 0 - num_devices, 1 - num_loop, 2 - [time,energy,reward]
    
    # algorithms = ['User Device', 'Edge', 'Vertical HEFT', 'Vertical CPOP', 'Vertical PEFT', 'Vertical PSO-GA', 'Vertical Genetic', 'Hybrid CPOP', 'Hybrid PEFT', 'Hybrid HEFT', 'Hybrid PSO-GA', 'Hybrid Genetic']
    algorithms = ['Vertical HEFT', 'Hybrid HEFT', 'Hybrid Heuristic']
    graph_type = 'line'

    # hatch = {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    if graph_type == 'bar':
        styles = [
            {'color':'white', 'edgecolor':'black', 'hatch':'///' },
            {'color':'deepskyblue', 'edgecolor':'black', 'hatch':'\\\\\\' },
            {'color':'yellow', 'edgecolor':'black', 'hatch':'|||' },
            {'color':'tomato', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'lime', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'orange', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'purple', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'red', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'blue', 'edgecolor':'black', 'hatch':'xxx' },
            {'color':'green', 'edgecolor':'black', 'hatch':'xxx' },
        ]
        
    elif graph_type == 'line':
        styles = [
            # {'c':'red', 'linestyle':'--', 'marker':'x', 'markersize':6},
            # {'c':'blue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            # {'c':'green', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'deeppink', 'linestyle':'--', 'marker':'x', 'markersize':6},
            {'c':'dodgerblue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            {'c':'limegreen', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'orange', 'linestyle':'--', 'marker':'s', 'markersize':6},
        ]

    x_range = ["0.1", "0.5", "1.0", "2.0", "3.0"]
    x = [x for x in x_range]
    y = []
    plt.figure(figsize=(8,5))
    plt.title("Number of IoT Devices = {}, Number of Services = {}".format(num_servers, num_services))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services <= 3:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services <= 6:
        plt.yticks(np.arange(0, 1000+1, 50))
    else:
        plt.yticks(np.arange(0, 1000+1, 100))
    plt.ylim(0, np.max(np.array(heft_result)[:,:,0], axis=None) * 1200)
    # plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,0] * 1000
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,0] * 1000
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,0] * 1000
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,0] * 1000
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,0] * 1000
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,0] * 1000
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,0] * 1000
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,0] * 1000
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "latency", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
            # for index in range(len(x)):
            #     plt.text(x[index], np.round(avg, 2)[index], np.round(avg, 2)[index], size=8)
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Communication-to-computation Ratio(CCR)')
    plt.ylabel('Total completion time (ms)')
    plt.savefig(os.path.join(dir, 'outputs', 'delay_bandwidth_D={}_M={}.png'.format(num_servers, num_services)))
    plt.clf()

    y = []
    plt.title("Number of IoT Devices = {}, Number of Services = {}".format(num_servers, num_services))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 200+1, 1))
    elif num_services < 5:
        plt.yticks(np.arange(0, 200+1, 2))
    else:
        plt.yticks(np.arange(0, 200+1, 10))
    plt.ylim(0, np.max(np.array(heft_result)[:,:,1], axis=None) * 1.1)
    # plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,1]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,1]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,1]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,1]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,1]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,1]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,1]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,1]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,1]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,1]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,1]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,1]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,1]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "energy", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Communication-to-computation Ratio(CCR)')
    plt.ylabel('IoT Energy Consumption (mJ)')
    plt.savefig(os.path.join(dir, 'outputs', 'energy_bandwidth_D={}_M={}.png'.format(num_servers, num_services)))
    plt.clf()

    y = []
    plt.title("Number of IoT Devices = {}, Number of Services = {}".format(num_servers, num_services))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(-1, 1+1, 0.01))
    elif num_services < 4:
        plt.yticks(np.arange(-1, 1+1, 0.02))
    else:
        plt.yticks(np.arange(-1, 1+1, 0.05))
    plt.ylim(np.min(np.array(heft_result)[:,:,2], axis=None) * 1.1, 0)
    # plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,2]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,2]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_result)[:,:,2]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_result)[:,:,2]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_result)[:,:,2]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_result)[:,:,2]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_result)[:,:,2]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,2]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_result)[:,:,2]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_result)[:,:,2]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_result)[:,:,2]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_result)[:,:,2]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "reward", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Communication-to-computation Ratio(CCR)')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(dir, 'outputs', 'reward_bandwidth_D={}_M={}.png'.format(num_servers, num_services)))
    plt.clf()

    algorithms = ['Vertical HEFT', 'Hybrid HEFT', 'Hybrid Heuristic']
    y = []
    plt.title("Number of IoT Devices = {}, Number of Services = {}".format(num_servers, num_services))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 1000+1, 1))
    elif num_services < 4:
        plt.yticks(np.arange(0, 1000+1, 5))
    elif num_services < 5:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services < 7:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services < 8:
        plt.yticks(np.arange(0, 1000+1, 20))
    else:
        plt.yticks(np.arange(0, 1000+1, 20))
    plt.ylim(0, max(np.mean(greedy_took_lst, axis=1)) * 1.2)
    # plt.xticks(x_range)
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_took_lst)[:,:]
        elif algorithm == 'Edge':
            avg = np.array(edge_took_lst)[:,:]
        elif algorithm == 'Hybrid HEFT':
            avg = np.array(heft_took_lst)[:,:]
        elif algorithm == 'Hybrid CPOP':
            avg = np.array(cpop_took_lst)[:,:]
        elif algorithm == 'Hybrid PEFT':
            avg = np.array(peft_took_lst)[:,:]
        elif algorithm == 'Hybrid Heuristic':
            avg = np.array(greedy_took_lst)[:,:]
        elif algorithm == 'Vertical HEFT':
            avg = np.array(layerwise_heft_took_lst)[:,:]
        elif algorithm == 'Vertical CPOP':
            avg = np.array(layerwise_cpop_took_lst)[:,:]
        elif algorithm == 'Vertical PEFT':
            avg = np.array(layerwise_peft_took_lst)[:,:]
        elif algorithm == 'Vertical Genetic':
            avg = np.array(layerwise_genetic_took_lst)[:,:]
        elif algorithm == 'Vertical PSO-GA':
            avg = np.array(layerwise_psoga_took_lst)[:,:]
        elif algorithm == 'Hybrid Genetic':
            avg = np.array(genetic_took_lst)[:,:]
        elif algorithm == 'Hybrid PSO-GA':
            avg = np.array(psoga_took_lst)[:,:]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "time", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Communication-to-computation Ratio(CCR)')
    plt.ylabel('Execution Time (sec)')
    plt.savefig(os.path.join(dir, 'outputs', 'execution_time_bandwidth_D={}_M={}.png'.format(num_servers, num_services)))
    plt.clf()

def compute_pos(xticks, width, i, models):
    index = np.arange(len(xticks))
    n = len(models)
    correction = i - 0.5 * (n - 1)
    return index + width * correction

def present_height(ax, bar):
    for rect in bar:
        height = rect.get_height()
        posx = rect.get_x() + rect.get_width() * 0.5
        posy = height * 1.01
        ax.text(posx, posy, '%.3f' % height, rotation=90, ha='center', va='bottom')

if __name__ == '__main__':
    import pickle
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
    server_low = 0
    server_high = 10
    server_step = 2
    num_services = 6
    bandwidth = 1.0
    for num_servers in range(server_low, server_high+1, server_step):
        with open("outputs/results_backup_service_{}_server_{}_bandwidth_{}".format(num_services, num_servers, bandwidth), "rb") as fp:
            temp_local_result, temp_edge_result, temp_heft_result, temp_cpop_result, temp_peft_result, temp_greedy_result, temp_layerwise_heft_result, temp_layerwise_cpop_result, temp_layerwise_peft_result, temp_layerwise_psoga_result, temp_layerwise_genetic_result, temp_psoga_result, temp_genetic_result, temp_layerwise_psoga_eval_lst, temp_layerwise_genetic_eval_lst, temp_psoga_eval_lst, temp_genetic_eval_lst, temp_local_took_lst, temp_edge_took_lst, temp_heft_took_lst, temp_cpop_took_lst, temp_peft_took_lst, temp_greedy_took_lst, temp_layerwise_heft_took_lst, temp_layerwise_cpop_took_lst, temp_layerwise_peft_took_lst, temp_layerwise_psoga_took_lst, temp_layerwise_genetic_took_lst, temp_psoga_took_lst, temp_genetic_took_lst, temp_local_action_lst, temp_edge_action_lst, temp_heft_action_lst, temp_cpop_action_lst, temp_peft_action_lst, temp_greedy_action_lst, temp_layerwise_heft_action_lst, temp_layerwise_cpop_action_lst, temp_layerwise_peft_action_lst, temp_layerwise_psoga_action_lst, temp_layerwise_genetic_action_lst, temp_psoga_action_lst, temp_genetic_action_lst, temp_server_low, temp_server_high, temp_server_step, temp_num_services = pickle.load(fp)
            local_result.extend(temp_local_result)
            edge_result.extend(temp_edge_result)
            heft_result.extend(temp_heft_result)
            cpop_result.extend(temp_cpop_result)
            peft_result.extend(temp_peft_result)
            greedy_result.extend(temp_greedy_result)
            layerwise_heft_result.extend(temp_layerwise_heft_result)
            layerwise_cpop_result.extend(temp_layerwise_cpop_result)
            layerwise_peft_result.extend(temp_layerwise_peft_result)
            layerwise_psoga_result.extend(temp_layerwise_psoga_result)
            layerwise_genetic_result.extend(temp_layerwise_genetic_result)
            psoga_result.extend(temp_psoga_result)
            genetic_result.extend(temp_genetic_result)
            local_took_lst.extend(temp_local_took_lst)
            edge_took_lst.extend(temp_edge_took_lst)
            heft_took_lst.extend(temp_heft_took_lst)
            cpop_took_lst.extend(temp_cpop_took_lst)
            peft_took_lst.extend(temp_peft_took_lst)
            greedy_took_lst.extend(temp_greedy_took_lst)
            layerwise_heft_took_lst.extend(temp_layerwise_heft_took_lst)
            layerwise_cpop_took_lst.extend(temp_layerwise_cpop_took_lst)
            layerwise_peft_took_lst.extend(temp_layerwise_peft_took_lst)
            layerwise_psoga_took_lst.extend(temp_layerwise_psoga_took_lst)
            layerwise_genetic_took_lst.extend(temp_layerwise_genetic_took_lst)
            psoga_took_lst.extend(temp_psoga_took_lst)
            genetic_took_lst.extend(temp_genetic_took_lst)
            local_action_lst.extend(temp_local_action_lst)
            edge_action_lst.extend(temp_edge_action_lst)
            heft_action_lst.extend(temp_heft_action_lst)
            cpop_action_lst.extend(temp_cpop_action_lst)
            peft_action_lst.extend(temp_peft_action_lst)
            greedy_action_lst.extend(temp_greedy_action_lst)
            layerwise_heft_action_lst.extend(temp_layerwise_heft_action_lst)
            layerwise_cpop_action_lst.extend(temp_layerwise_cpop_action_lst)
            layerwise_peft_action_lst.extend(temp_layerwise_peft_action_lst)
            layerwise_psoga_action_lst.extend(temp_layerwise_psoga_action_lst)
            layerwise_genetic_action_lst.extend(temp_layerwise_genetic_action_lst)
            psoga_action_lst.extend(temp_psoga_action_lst)
            genetic_action_lst.extend(temp_genetic_action_lst)
    by_num_service(local_result, edge_result, heft_result, cpop_result, peft_result, greedy_result, layerwise_heft_result, layerwise_cpop_result, layerwise_peft_result, layerwise_psoga_result, layerwise_genetic_result, psoga_result, genetic_result, layerwise_psoga_eval_lst, layerwise_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, local_took_lst, edge_took_lst, heft_took_lst, cpop_took_lst, peft_took_lst, greedy_took_lst, layerwise_heft_took_lst, layerwise_cpop_took_lst, layerwise_peft_took_lst, layerwise_psoga_took_lst, layerwise_genetic_took_lst, psoga_took_lst, genetic_took_lst, local_action_lst, edge_action_lst, heft_action_lst, cpop_action_lst, peft_action_lst, greedy_action_lst, layerwise_heft_action_lst, layerwise_cpop_action_lst, layerwise_peft_action_lst, layerwise_psoga_action_lst, layerwise_genetic_action_lst, psoga_action_lst, genetic_action_lst, server_low, server_high, server_step, num_services, bandwidth)
    
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
    service_low = 1
    service_high = 9
    service_step = 1
    num_servers = 6
    bandwidth = 1.0
    for num_services in range(service_low, service_high+1, service_step):
        with open("outputs/results_backup_service_{}_server_{}_bandwidth_{}".format(num_services, num_servers, bandwidth), "rb") as fp:
            temp_local_result, temp_edge_result, temp_heft_result, temp_cpop_result, temp_peft_result, temp_greedy_result, temp_layerwise_heft_result, temp_layerwise_cpop_result, temp_layerwise_peft_result, temp_layerwise_psoga_result, temp_layerwise_genetic_result, temp_psoga_result, temp_genetic_result, temp_layerwise_psoga_eval_lst, temp_layerwise_genetic_eval_lst, temp_psoga_eval_lst, temp_genetic_eval_lst, temp_local_took_lst, temp_edge_took_lst, temp_heft_took_lst, temp_cpop_took_lst, temp_peft_took_lst, temp_greedy_took_lst, temp_layerwise_heft_took_lst, temp_layerwise_cpop_took_lst, temp_layerwise_peft_took_lst, temp_layerwise_psoga_took_lst, temp_layerwise_genetic_took_lst, temp_psoga_took_lst, temp_genetic_took_lst, temp_local_action_lst, temp_edge_action_lst, temp_heft_action_lst, temp_cpop_action_lst, temp_peft_action_lst, temp_greedy_action_lst, temp_layerwise_heft_action_lst, temp_layerwise_cpop_action_lst, temp_layerwise_peft_action_lst, temp_layerwise_psoga_action_lst, temp_layerwise_genetic_action_lst, temp_psoga_action_lst, temp_genetic_action_lst, temp_server_low, temp_server_high, temp_server_step, temp_num_services = pickle.load(fp)
            local_result.extend(temp_local_result)
            edge_result.extend(temp_edge_result)
            heft_result.extend(temp_heft_result)
            cpop_result.extend(temp_cpop_result)
            peft_result.extend(temp_peft_result)
            greedy_result.extend(temp_greedy_result)
            layerwise_heft_result.extend(temp_layerwise_heft_result)
            layerwise_cpop_result.extend(temp_layerwise_cpop_result)
            layerwise_peft_result.extend(temp_layerwise_peft_result)
            layerwise_psoga_result.extend(temp_layerwise_psoga_result)
            layerwise_genetic_result.extend(temp_layerwise_genetic_result)
            psoga_result.extend(temp_psoga_result)
            genetic_result.extend(temp_genetic_result)
            local_took_lst.extend(temp_local_took_lst)
            edge_took_lst.extend(temp_edge_took_lst)
            heft_took_lst.extend(temp_heft_took_lst)
            cpop_took_lst.extend(temp_cpop_took_lst)
            peft_took_lst.extend(temp_peft_took_lst)
            greedy_took_lst.extend(temp_greedy_took_lst)
            layerwise_heft_took_lst.extend(temp_layerwise_heft_took_lst)
            layerwise_cpop_took_lst.extend(temp_layerwise_cpop_took_lst)
            layerwise_peft_took_lst.extend(temp_layerwise_peft_took_lst)
            layerwise_psoga_took_lst.extend(temp_layerwise_psoga_took_lst)
            layerwise_genetic_took_lst.extend(temp_layerwise_genetic_took_lst)
            psoga_took_lst.extend(temp_psoga_took_lst)
            genetic_took_lst.extend(temp_genetic_took_lst)
            local_action_lst.extend(temp_local_action_lst)
            edge_action_lst.extend(temp_edge_action_lst)
            heft_action_lst.extend(temp_heft_action_lst)
            cpop_action_lst.extend(temp_cpop_action_lst)
            peft_action_lst.extend(temp_peft_action_lst)
            greedy_action_lst.extend(temp_greedy_action_lst)
            layerwise_heft_action_lst.extend(temp_layerwise_heft_action_lst)
            layerwise_cpop_action_lst.extend(temp_layerwise_cpop_action_lst)
            layerwise_peft_action_lst.extend(temp_layerwise_peft_action_lst)
            layerwise_psoga_action_lst.extend(temp_layerwise_psoga_action_lst)
            layerwise_genetic_action_lst.extend(temp_layerwise_genetic_action_lst)
            psoga_action_lst.extend(temp_psoga_action_lst)
            genetic_action_lst.extend(temp_genetic_action_lst)
    by_num_servers(local_result, edge_result, heft_result, cpop_result, peft_result, greedy_result, layerwise_heft_result, layerwise_cpop_result, layerwise_peft_result, layerwise_psoga_result, layerwise_genetic_result, psoga_result, genetic_result, layerwise_psoga_eval_lst, layerwise_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, local_took_lst, edge_took_lst, heft_took_lst, cpop_took_lst, peft_took_lst, greedy_took_lst, layerwise_heft_took_lst, layerwise_cpop_took_lst, layerwise_peft_took_lst, layerwise_psoga_took_lst, layerwise_genetic_took_lst, psoga_took_lst, genetic_took_lst, local_action_lst, edge_action_lst, heft_action_lst, cpop_action_lst, peft_action_lst, greedy_action_lst, layerwise_heft_action_lst, layerwise_cpop_action_lst, layerwise_peft_action_lst, layerwise_psoga_action_lst, layerwise_genetic_action_lst, psoga_action_lst, genetic_action_lst, service_low, service_high, service_step, num_servers, bandwidth)
    
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
    num_services = 6
    num_servers = 6
    for bandwidth in [0.1, 0.5, 1.0, 2.0, 3.0]:
        with open("outputs/results_backup_service_{}_server_{}_bandwidth_{}".format(num_services, num_servers, bandwidth), "rb") as fp:
            temp_local_result, temp_edge_result, temp_heft_result, temp_cpop_result, temp_peft_result, temp_greedy_result, temp_layerwise_heft_result, temp_layerwise_cpop_result, temp_layerwise_peft_result, temp_layerwise_psoga_result, temp_layerwise_genetic_result, temp_psoga_result, temp_genetic_result, temp_layerwise_psoga_eval_lst, temp_layerwise_genetic_eval_lst, temp_psoga_eval_lst, temp_genetic_eval_lst, temp_local_took_lst, temp_edge_took_lst, temp_heft_took_lst, temp_cpop_took_lst, temp_peft_took_lst, temp_greedy_took_lst, temp_layerwise_heft_took_lst, temp_layerwise_cpop_took_lst, temp_layerwise_peft_took_lst, temp_layerwise_psoga_took_lst, temp_layerwise_genetic_took_lst, temp_psoga_took_lst, temp_genetic_took_lst, temp_local_action_lst, temp_edge_action_lst, temp_heft_action_lst, temp_cpop_action_lst, temp_peft_action_lst, temp_greedy_action_lst, temp_layerwise_heft_action_lst, temp_layerwise_cpop_action_lst, temp_layerwise_peft_action_lst, temp_layerwise_psoga_action_lst, temp_layerwise_genetic_action_lst, temp_psoga_action_lst, temp_genetic_action_lst, temp_server_low, temp_server_high, temp_server_step, temp_num_services = pickle.load(fp)
            local_result.extend(temp_local_result)
            edge_result.extend(temp_edge_result)
            heft_result.extend(temp_heft_result)
            cpop_result.extend(temp_cpop_result)
            peft_result.extend(temp_peft_result)
            greedy_result.extend(temp_greedy_result)
            layerwise_heft_result.extend(temp_layerwise_heft_result)
            layerwise_cpop_result.extend(temp_layerwise_cpop_result)
            layerwise_peft_result.extend(temp_layerwise_peft_result)
            layerwise_psoga_result.extend(temp_layerwise_psoga_result)
            layerwise_genetic_result.extend(temp_layerwise_genetic_result)
            psoga_result.extend(temp_psoga_result)
            genetic_result.extend(temp_genetic_result)
            local_took_lst.extend(temp_local_took_lst)
            edge_took_lst.extend(temp_edge_took_lst)
            heft_took_lst.extend(temp_heft_took_lst)
            cpop_took_lst.extend(temp_cpop_took_lst)
            peft_took_lst.extend(temp_peft_took_lst)
            greedy_took_lst.extend(temp_greedy_took_lst)
            layerwise_heft_took_lst.extend(temp_layerwise_heft_took_lst)
            layerwise_cpop_took_lst.extend(temp_layerwise_cpop_took_lst)
            layerwise_peft_took_lst.extend(temp_layerwise_peft_took_lst)
            layerwise_psoga_took_lst.extend(temp_layerwise_psoga_took_lst)
            layerwise_genetic_took_lst.extend(temp_layerwise_genetic_took_lst)
            psoga_took_lst.extend(temp_psoga_took_lst)
            genetic_took_lst.extend(temp_genetic_took_lst)
            local_action_lst.extend(temp_local_action_lst)
            edge_action_lst.extend(temp_edge_action_lst)
            heft_action_lst.extend(temp_heft_action_lst)
            cpop_action_lst.extend(temp_cpop_action_lst)
            peft_action_lst.extend(temp_peft_action_lst)
            greedy_action_lst.extend(temp_greedy_action_lst)
            layerwise_heft_action_lst.extend(temp_layerwise_heft_action_lst)
            layerwise_cpop_action_lst.extend(temp_layerwise_cpop_action_lst)
            layerwise_peft_action_lst.extend(temp_layerwise_peft_action_lst)
            layerwise_psoga_action_lst.extend(temp_layerwise_psoga_action_lst)
            layerwise_genetic_action_lst.extend(temp_layerwise_genetic_action_lst)
            psoga_action_lst.extend(temp_psoga_action_lst)
            genetic_action_lst.extend(temp_genetic_action_lst)
    by_num_bandwidths(local_result, edge_result, heft_result, cpop_result, peft_result, greedy_result, layerwise_heft_result, layerwise_cpop_result, layerwise_peft_result, layerwise_psoga_result, layerwise_genetic_result, psoga_result, genetic_result, layerwise_psoga_eval_lst, layerwise_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, local_took_lst, edge_took_lst, heft_took_lst, cpop_took_lst, peft_took_lst, greedy_took_lst, layerwise_heft_took_lst, layerwise_cpop_took_lst, layerwise_peft_took_lst, layerwise_psoga_took_lst, layerwise_genetic_took_lst, psoga_took_lst, genetic_took_lst, local_action_lst, edge_action_lst, heft_action_lst, cpop_action_lst, peft_action_lst, greedy_action_lst, layerwise_heft_action_lst, layerwise_cpop_action_lst, layerwise_peft_action_lst, layerwise_psoga_action_lst, layerwise_genetic_action_lst, psoga_action_lst, genetic_action_lst, num_services, num_servers, bandwidth)