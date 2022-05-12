import os
import numpy as np
from matplotlib import pyplot as plt

# Get results by the number of services
def by_num_service(local_result, edge_result, greedy_result, memetic_result, psoga_result, genetic_result, memetic_eval_lst, psoga_eval_lst, genetic_eval_lst, server_low, server_high, server_step, num_services):
    # [0,1,2]: 0 - num_devices, 1 - num_loop, 2 - [time,energy,reward]
    
    algorithms = ['Greedy', 'PSO-GA']
    styles = [
        {'c':'purple', 'linestyle':'--', 'marker':'^', 'markersize':7},
        {'c':'green', 'linestyle':'--', 'marker':'s', 'markersize':7},
        {'c':'blue', 'linestyle':'--', 'marker':'o', 'markersize':7},
        {'c':'red', 'linestyle':'--', 'marker':'x', 'markersize':7},
    ]
    x = [x for x in range(server_low, server_high+1, server_step)]
    y = []
    plt.figure(figsize=(8,4))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,0,0] * 1000
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,0,0] * 1000
        elif algorithm == 'Greedy':
            avg = np.array(greedy_result)[:,0,0] * 1000
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,0,0] * 1000
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,0,0] * 1000
        elif algorithm == 'Memetic':
            avg = np.array(memetic_result)[:,0,0] * 1000
        else:
            RuntimeError("Algorithm not matching!")
        print(algorithm, "latency", avg)
        y.append(avg)
        plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of local devices'.format(num_services))
    plt.ylabel('Total completion time (ms)')
    plt.savefig(os.path.join(dir, 'outputs', 'delay_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,0,1]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,0,1]
        elif algorithm == 'Greedy':
            avg = np.array(greedy_result)[:,0,1]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,0,1]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,0,1]
        elif algorithm == 'Memetic':
            avg = np.array(memetic_result)[:,0,1]
        else:
            RuntimeError("Algorithm not matching!")
        print(algorithm, "energy", avg)
        y.append(avg)
        plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of local devices'.format(num_services))
    plt.ylabel('Total Energy Consumption')
    plt.savefig(os.path.join(dir, 'outputs', 'energy_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,0,2]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,0,2]
        elif algorithm == 'Greedy':
            avg = np.array(greedy_result)[:,0,2]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,0,2]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,0,2]
        elif algorithm == 'Memetic':
            avg = np.array(memetic_result)[:,0,2]
        else:
            RuntimeError("Algorithm not matching!")
        print(algorithm, "reward", avg)
        y.append(avg)
        plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of local devices'.format(num_services))
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(dir, 'outputs', 'reward_devices_M={}.png'.format(num_services)))
    plt.clf()


if __name__ == '__main__':
    import pickle
    with open("outputs copy/results_backup", "rb") as fp:
        result_by_services = pickle.load(fp)
    for result_by_servers in result_by_services:
        by_num_service(*result_by_servers)