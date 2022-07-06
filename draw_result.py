import os
import numpy as np
from matplotlib import pyplot as plt

# Get results by the number of services
def by_num_service(local_result, edge_result, heft_u_result, heft_d_result, cpop_result, peft_result, layerwise_heft_u_result, layerwise_heft_d_result, layerwise_cpop_result, layerwise_peft_result, greedy_u_result, greedy_d_result, greedy_psoga_result, greedy_genetic_result, psoga_result, genetic_result, greedy_psoga_eval_lst, greedy_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, greedy_psoga_took_lst, greedy_genetic_took_lst, psoga_took_lst, genetic_took_lst, server_low, server_high, server_step, num_services):
    # [0,1,2]: 0 - num_devices, 1 - num_loop, 2 - [time,energy,reward]
    
    # algorithms = ['User Device', 'Edge', 'CPOP', 'PEFT', 'HEFT-U', 'HEFT-D', 'Greedy-U', 'Greedy-D', 'Genetic', 'PSO-GA']
    algorithms = ['CPOP', 'PEFT', 'HEFT-U', 'Genetic']
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
            {'c':'deeppink', 'linestyle':'--', 'marker':'x', 'markersize':6},
            {'c':'dodgerblue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            {'c':'purple', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'orange', 'linestyle':'--', 'marker':'s', 'markersize':6},

            {'c':'red', 'linestyle':'--', 'marker':'x', 'markersize':6},
            {'c':'green', 'linestyle':'--', 'marker':'s', 'markersize':6},
            {'c':'blue', 'linestyle':'--', 'marker':'o', 'markersize':6},
        ]

    x = [x for x in range(server_low, server_high+1, server_step)]
    y = []
    plt.figure(figsize=(8,6))
    plt.title("Number of DNN = {}".format(num_services))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services <= 3:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services <= 6:
        plt.yticks(np.arange(0, 1000+1, 50))
    else:
        plt.yticks(np.arange(0, 1000+1, 100))
    plt.ylim(0, max(np.max(np.array(heft_u_result)[:,:,0], axis=None), np.max(np.array(cpop_result)[:,:,0], axis=None), np.max(np.array(peft_result)[:,:,0], axis=None)) * 1200)
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,0] * 1000
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,0] * 1000
        elif algorithm == 'HEFT-U':
            avg = np.array(heft_u_result)[:,:,0] * 1000
        elif algorithm == 'HEFT-D':
            avg = np.array(heft_d_result)[:,:,0] * 1000
        elif algorithm == 'CPOP':
            avg = np.array(cpop_result)[:,:,0] * 1000
        elif algorithm == 'PEFT':
            avg = np.array(peft_result)[:,:,0] * 1000
        elif algorithm == 'Layerwise HEFT-U':
            avg = np.array(layerwise_heft_u_result)[:,:,0] * 1000
        elif algorithm == 'Layerwise HEFT-D':
            avg = np.array(layerwise_heft_d_result)[:,:,0] * 1000
        elif algorithm == 'Layerwise CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,0] * 1000
        elif algorithm == 'Layerwise PEFT':
            avg = np.array(layerwise_peft_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-U':
            avg = np.array(greedy_u_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-D':
            avg = np.array(greedy_d_result)[:,:,0] * 1000
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,0] * 1000
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-Genetic':
            avg = np.array(greedy_psoga_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-PSOGA':
            avg = np.array(greedy_psoga_result)[:,:,0] * 1000
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "latency", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
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
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 200+1, 1))
    elif num_services < 5:
        plt.yticks(np.arange(0, 200+1, 2))
    else:
        plt.yticks(np.arange(0, 200+1, 5))
    plt.ylim(0, max(np.max(np.array(heft_u_result)[:,:,1], axis=None), np.max(np.array(cpop_result)[:,:,1], axis=None), np.max(np.array(peft_result)[:,:,1], axis=None)) * 1.1)
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,1]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,1]
        elif algorithm == 'HEFT-U':
            avg = np.array(heft_u_result)[:,:,1]
        elif algorithm == 'HEFT-D':
            avg = np.array(heft_d_result)[:,:,1]
        elif algorithm == 'CPOP':
            avg = np.array(cpop_result)[:,:,1]
        elif algorithm == 'PEFT':
            avg = np.array(peft_result)[:,:,1]
        elif algorithm == 'Layerwise HEFT-U':
            avg = np.array(layerwise_heft_u_result)[:,:,1]
        elif algorithm == 'Layerwise HEFT-D':
            avg = np.array(layerwise_heft_d_result)[:,:,1]
        elif algorithm == 'Layerwise CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,1]
        elif algorithm == 'Layerwise PEFT':
            avg = np.array(layerwise_peft_result)[:,:,1]
        elif algorithm == 'Greedy-U':
            avg = np.array(greedy_u_result)[:,:,1]
        elif algorithm == 'Greedy-D':
            avg = np.array(greedy_d_result)[:,:,1]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,1]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,1]
        elif algorithm == 'Greedy-Genetic':
            avg = np.array(greedy_psoga_result)[:,:,1]
        elif algorithm == 'Greedy-PSOGA':
            avg = np.array(greedy_psoga_result)[:,:,1]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "energy", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Total Energy Consumption')
    plt.savefig(os.path.join(dir, 'outputs', 'energy_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(-1, 1+1, 0.01))
    elif num_services < 4:
        plt.yticks(np.arange(-1, 1+1, 0.02))
    else:
        plt.yticks(np.arange(-1, 1+1, 0.05))
    plt.ylim(min(np.min(np.array(heft_u_result)[:,:,2], axis=None), np.min(np.array(cpop_result)[:,:,2], axis=None), np.min(np.array(peft_result)[:,:,2], axis=None)) * 1.1, 0)
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,2]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,2]
        elif algorithm == 'HEFT-U':
            avg = np.array(heft_u_result)[:,:,2]
        elif algorithm == 'HEFT-D':
            avg = np.array(heft_d_result)[:,:,2]
        elif algorithm == 'CPOP':
            avg = np.array(cpop_result)[:,:,2]
        elif algorithm == 'PEFT':
            avg = np.array(peft_result)[:,:,2]
        elif algorithm == 'Layerwise HEFT-U':
            avg = np.array(layerwise_heft_u_result)[:,:,2]
        elif algorithm == 'Layerwise HEFT-D':
            avg = np.array(layerwise_heft_d_result)[:,:,2]
        elif algorithm == 'Layerwise CPOP':
            avg = np.array(layerwise_cpop_result)[:,:,2]
        elif algorithm == 'Layerwise PEFT':
            avg = np.array(layerwise_peft_result)[:,:,2]
        elif algorithm == 'Greedy-U':
            avg = np.array(greedy_u_result)[:,:,2]
        elif algorithm == 'Greedy-D':
            avg = np.array(greedy_d_result)[:,:,2]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,2]
        elif algorithm == 'Greedy-Genetic':
            avg = np.array(greedy_psoga_result)[:,:,2]
        elif algorithm == 'Greedy-PSOGA':
            avg = np.array(greedy_psoga_result)[:,:,2]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "reward", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(dir, 'outputs', 'reward_devices_M={}.png'.format(num_services)))
    plt.clf()

    algorithms = ['Genetic']
    y = []
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 1000+1, 1))
    elif num_services < 4:
        plt.yticks(np.arange(0, 1000+1, 5))
    elif num_services < 5:
        plt.yticks(np.arange(0, 1000+1, 10))
    elif num_services < 7:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services < 8:
        plt.yticks(np.arange(0, 1000+1, 50))
    else:
        plt.yticks(np.arange(0, 1000+1, 100))
    plt.ylim(0, max(np.mean(genetic_took_lst, axis=1)) * 1.2)
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'Genetic':
            avg = np.array(genetic_took_lst)[:,:]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_took_lst)[:,:]
        elif algorithm == 'Greedy-Genetic':
            avg = np.array(greedy_genetic_took_lst)[:,:]
        elif algorithm == 'Greedy-PSOGA':
            avg = np.array(greedy_psoga_took_lst)[:,:]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "time", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Execution Time')
    plt.savefig(os.path.join(dir, 'outputs', 'execution_time_devices_M={}.png'.format(num_services)))
    plt.clf()

# Get results by the number of services
def by_num_service_test(local_result, edge_result, heft_u_result, heft_d_result, cpop_result, peft_result, layerwise_heft_u_result, layerwise_heft_d_result, layerwise_cpop_result, layerwise_peft_result, greedy_u_result, greedy_d_result, greedy_psoga_result, greedy_genetic_result, psoga_result, genetic_result, greedy_psoga_eval_lst, greedy_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, greedy_psoga_took_lst, greedy_genetic_took_lst, psoga_took_lst, genetic_took_lst, server_low, server_high, server_step, num_services):
    # [0,1,2]: 0 - num_devices, 1 - num_loop, 2 - [time,energy,reward]
    
    # algorithms = ['User Device', 'Edge', 'CPOP', 'PEFT', 'HEFT-U', 'HEFT-D', 'Greedy-U', 'Greedy-D', 'Genetic', 'PSO-GA']
    algorithms = ['CPOP', 'PEFT', 'HEFT-U']
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
            {'c':'deeppink', 'linestyle':'--', 'marker':'x', 'markersize':6},
            {'c':'dodgerblue', 'linestyle':'--', 'marker':'o', 'markersize':6},
            {'c':'purple', 'linestyle':'--', 'marker':'^', 'markersize':6},
            {'c':'orange', 'linestyle':'--', 'marker':'s', 'markersize':6},

            {'c':'red', 'linestyle':'--', 'marker':'x', 'markersize':6},
            {'c':'green', 'linestyle':'--', 'marker':'s', 'markersize':6},
            {'c':'blue', 'linestyle':'--', 'marker':'o', 'markersize':6},
        ]

    x = [x for x in range(server_low, server_high+1, server_step)]
    y = []
    plt.figure(figsize=(8,6))
    plt.title("Number of DNN = {}".format(num_services))
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services <= 3:
        plt.yticks(np.arange(0, 1000+1, 20))
    elif num_services <= 6:
        plt.yticks(np.arange(0, 1000+1, 50))
    else:
        plt.yticks(np.arange(0, 1000+1, 100))
    plt.ylim(0, np.mean(np.array(edge_result)[:,:,0], axis=None) * 1200)
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,0] * 1000
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,0] * 1000
        elif algorithm == 'HEFT-U':
            avg = np.array(heft_u_result)[:,:,0] * 1000
        elif algorithm == 'HEFT-D':
            avg = np.array(heft_d_result)[:,:,0] * 1000
        elif algorithm == 'CPOP':
            avg = np.array(cpop_result)[:,:,0] * 1000
        elif algorithm == 'PEFT':
            avg = np.array(peft_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-U':
            avg = np.array(greedy_u_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-D':
            avg = np.array(greedy_d_result)[:,:,0] * 1000
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,0] * 1000
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-Genetic':
            avg = np.array(greedy_psoga_result)[:,:,0] * 1000
        elif algorithm == 'Greedy-PSOGA':
            avg = np.array(greedy_psoga_result)[:,:,0] * 1000
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "latency", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
            # for index in range(len(x)):
            #     plt.text(x[index], np.round(avg, 2)[index], np.round(avg, 2)[index], size=8)
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Total completion time (ms)')
    plt.savefig(os.path.join(dir, 'outputs', 'test_delay_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(0, 200+1, 1))
    elif num_services < 5:
        plt.yticks(np.arange(0, 200+1, 2))
    else:
        plt.yticks(np.arange(0, 200+1, 5))
    plt.ylim(0, max(np.max(np.array(heft_u_result)[:,:,1], axis=None), np.max(np.array(cpop_result)[:,:,1], axis=None), np.max(np.array(peft_result)[:,:,1], axis=None)) * 1.1)
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,1]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,1]
        elif algorithm == 'HEFT-U':
            avg = np.array(heft_u_result)[:,:,1]
        elif algorithm == 'HEFT-D':
            avg = np.array(heft_d_result)[:,:,1]
        elif algorithm == 'CPOP':
            avg = np.array(cpop_result)[:,:,1]
        elif algorithm == 'PEFT':
            avg = np.array(peft_result)[:,:,1]
        elif algorithm == 'Greedy-U':
            avg = np.array(greedy_u_result)[:,:,1]
        elif algorithm == 'Greedy-D':
            avg = np.array(greedy_d_result)[:,:,1]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,1]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,1]
        elif algorithm == 'Greedy-Genetic':
            avg = np.array(greedy_psoga_result)[:,:,1]
        elif algorithm == 'Greedy-PSOGA':
            avg = np.array(greedy_psoga_result)[:,:,1]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "energy", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Total Energy Consumption')
    plt.savefig(os.path.join(dir, 'outputs', 'test_energy_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    if num_services < 2:
        plt.yticks(np.arange(-1, 1+1, 0.01))
    elif num_services < 4:
        plt.yticks(np.arange(-1, 1+1, 0.02))
    else:
        plt.yticks(np.arange(-1, 1+1, 0.05))
    plt.ylim(min(np.min(np.array(heft_u_result)[:,:,2], axis=None), np.min(np.array(cpop_result)[:,:,2], axis=None), np.min(np.array(peft_result)[:,:,2], axis=None)) * 1.1, 0)
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,2]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,2]
        elif algorithm == 'HEFT-U':
            avg = np.array(heft_u_result)[:,:,2]
        elif algorithm == 'HEFT-D':
            avg = np.array(heft_d_result)[:,:,2]
        elif algorithm == 'CPOP':
            avg = np.array(cpop_result)[:,:,2]
        elif algorithm == 'PEFT':
            avg = np.array(peft_result)[:,:,2]
        elif algorithm == 'Greedy-U':
            avg = np.array(greedy_u_result)[:,:,2]
        elif algorithm == 'Greedy-D':
            avg = np.array(greedy_d_result)[:,:,2]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,2]
        elif algorithm == 'Greedy-Genetic':
            avg = np.array(greedy_psoga_result)[:,:,2]
        elif algorithm == 'Greedy-PSOGA':
            avg = np.array(greedy_psoga_result)[:,:,2]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "reward", avg)
        y.append(avg)

        if graph_type == 'bar':
            pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == 'line':
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices')
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(dir, 'outputs', 'test_reward_devices_M={}.png'.format(num_services)))
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
    with open("outputs/results_backup_9", "rb") as fp:
        result_by_services = pickle.load(fp)
    for result_by_servers in result_by_services:
        by_num_service(*result_by_servers)

    # with open("outputs/test_results_backup_{}".format(9), "rb") as fp:
    #     result_by_services = pickle.load(fp)
    # for result_by_servers in result_by_services:
    #     by_num_service_test(*result_by_servers)