import os
import numpy as np
from matplotlib import pyplot as plt

# Get results by the number of services
def by_num_service(local_result, edge_result, greedy_result, memetic_psoga_result, memetic_genetic_result, psoga_result, genetic_result, memetic_psoga_eval_lst, memetic_genetic_eval_lst, psoga_eval_lst, genetic_eval_lst, memetic_psoga_took_lst, memetic_genetic_took_lst, psoga_took_lst, genetic_took_lst, server_low, server_high, server_step, num_services):
    # [0,1,2]: 0 - num_devices, 1 - num_loop, 2 - [time,energy,reward]
    
    algorithms = ['Greedy', 'PSO-GA', 'Genetic', 'Memetic']

    # hatch = {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
    styles = [
        {'color':'white', 'edgecolor':'black', 'hatch':'///' },
        {'color':'deepskyblue', 'edgecolor':'black', 'hatch':'\\\\\\' },
        {'color':'yellow', 'edgecolor':'black', 'hatch':'|||' },
        {'color':'tomato', 'edgecolor':'black', 'hatch':'xxx' },
        {'color':'lime', 'edgecolor':'black', 'hatch':'xxx' },
        # {'c':'deeppink', 'linestyle':'--', 'marker':'x', 'markersize':7},
        # {'c':'dodgerblue', 'linestyle':'--', 'marker':'o', 'markersize':7},
        # {'c':'purple', 'linestyle':'--', 'marker':'^', 'markersize':7},
        # {'c':'green', 'linestyle':'--', 'marker':'s', 'markersize':7},
        # {'c':'blue', 'linestyle':'--', 'marker':'o', 'markersize':7},
        # {'c':'red', 'linestyle':'--', 'marker':'x', 'markersize':7},
    ]
    x = [x for x in range(server_low, server_high+1, server_step)]
    y = []
    plt.figure(figsize=(8,6))
    # plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    plt.yticks(np.arange(0, 5000+1, 100))
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,0] * 1000
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,0] * 1000
        elif algorithm == 'Greedy':
            avg = np.array(greedy_result)[:,:,0] * 1000
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,0] * 1000
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,0] * 1000
        elif algorithm == 'Memetic':
            avg = np.array(memetic_genetic_result)[:,:,0] * 1000
        elif algorithm == 'Memetic-PSO-GA':
            avg = np.array(memetic_psoga_result)[:,:,0] * 1000
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "latency", avg)
        y.append(avg)

        pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
        plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        # plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices'.format(num_services))
    plt.ylabel('Total completion time (ms)')
    plt.savefig(os.path.join(dir, 'outputs', 'delay_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    # plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    plt.yticks(np.arange(0, 100+1, 1))
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,1]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,1]
        elif algorithm == 'Greedy':
            avg = np.array(greedy_result)[:,:,1]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,1]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,1]
        elif algorithm == 'Memetic':
            avg = np.array(memetic_genetic_result)[:,:,1]
        elif algorithm == 'Memetic-PSO-GA':
            avg = np.array(memetic_psoga_result)[:,:,1]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "energy", avg)
        y.append(avg)

        pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
        plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        # plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices'.format(num_services))
    plt.ylabel('Total Energy Consumption')
    plt.savefig(os.path.join(dir, 'outputs', 'energy_devices_M={}.png'.format(num_services)))
    plt.clf()

    y = []
    # plt.grid(True, axis='both', color='gray', alpha=0.5, linestyle='--')
    plt.yticks(np.arange(0, 10, 0.2))
    plt.xticks(np.arange(0, 10+1, 2))
    for idx, algorithm in enumerate(algorithms):
        avg = None
        if algorithm == 'User Device':
            avg = np.array(local_result)[:,:,2]
        elif algorithm == 'Edge':
            avg = np.array(edge_result)[:,:,2]
        elif algorithm == 'Greedy':
            avg = np.array(greedy_result)[:,:,2]
        elif algorithm == 'Genetic':
            avg = np.array(genetic_result)[:,:,2]
        elif algorithm == 'PSO-GA':
            avg = np.array(psoga_result)[:,:,2]
        elif algorithm == 'Memetic':
            avg = np.array(memetic_genetic_result)[:,:,2]
        elif algorithm == 'Memetic-PSO-GA':
            avg = np.array(memetic_psoga_result)[:,:,2]
        else:
            RuntimeError("Algorithm not matching!")
        avg = np.mean(avg, axis=1)
        print(algorithm, "reward", avg)
        y.append(avg)

        pos = compute_pos(np.arange(server_low, server_high+1, server_step), 0.15, idx, algorithms)
        plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        # plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.legend()
    plt.xlabel('Number of IoT devices'.format(num_services))
    plt.ylabel('Total Reward')
    plt.savefig(os.path.join(dir, 'outputs', 'reward_devices_M={}.png'.format(num_services)))
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
    with open("outputs/results_backup", "rb") as fp:
        result_by_services = pickle.load(fp)
    for result_by_servers in result_by_services:
        by_num_service(*result_by_servers)