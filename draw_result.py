import os
import numpy as np
from matplotlib import pyplot as plt

# Get results by the number of servers
def by_num_servers(outputs, services, num_servers, bandwidth):
    partitioning = ["Piecewise", "Layerwise"]
    offloading = ["Local", "Edge", "HEFT", "CPOP", "PEFT", "Greedy", "PSOGA", "Genetic", "MemeticPSOGA", "MemeticGenetic"]
    algorithms = ["Layerwise Edge", "Layerwise PSOGA", "Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    rename_algorithms = {"Layerwise Edge": "Baseline", "Layerwise PSOGA": "LO", "Piecewise PSOGA": "PO", "Piecewise MemeticPSOGA": "MO"}
    graph_type = "line"

    # hatch = {"/", "\", "|", "-", "+", "x", "o", "O", ".", "*"}
    if graph_type == "bar":
        styles = [
            {"color":"white", "edgecolor":"black", "hatch":"///" },
            {"color":"deepskyblue", "edgecolor":"black", "hatch":"\\\\\\" },
            {"color":"yellow", "edgecolor":"black", "hatch":"|||" },
            {"color":"tomato", "edgecolor":"black", "hatch":"xxx" },
            {"color":"lime", "edgecolor":"black", "hatch":"xxx" },
            {"color":"orange", "edgecolor":"black", "hatch":"xxx" },
            {"color":"purple", "edgecolor":"black", "hatch":"xxx" },
            {"color":"red", "edgecolor":"black", "hatch":"xxx" },
            {"color":"blue", "edgecolor":"black", "hatch":"xxx" },
            {"color":"green", "edgecolor":"black", "hatch":"xxx" },
        ]
    elif graph_type == "line":
        styles = [
            {"c":"grey"},
            {"c":"red", "linestyle":"--", "marker":"x", "markersize":7, "linewidth":1},
            {"c":"blue", "linestyle":"--", "marker":"o", "markersize":7, "linewidth":1},
            {"c":"green", "linestyle":"--", "marker":"^", "markersize":7, "linewidth":1},
            {"c":"deeppink", "linestyle":"--", "marker":"x", "markersize":7, "linewidth":1},
            {"c":"dodgerblue", "linestyle":"--", "marker":"o", "markersize":7, "linewidth":1},
            {"c":"limegreen", "linestyle":"--", "marker":"^", "markersize":7, "linewidth":1},
            {"c":"orange", "linestyle":"--", "marker":"s", "markersize":7, "linewidth":1},
            {"c":"grey"},
        ]

    x_range = services
    x = [x for x in x_range]
    y = []
    plt.figure(figsize=(4,4))
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    plt.yticks(np.arange(0, 3+1, 0.1))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(result_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(result_lst)[:,:,0]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "latency", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx])
            # for index in range(len(x)):
            #     plt.text(x[index], np.round(avg, 2)[index], np.round(avg, 2)[index], size=8)
    dir = os.getcwd()
    
    plt.ylim(min_avg * 0.7, max_avg * 1.2)
    plt.legend(fontsize=12, loc="upper left")
    plt.xlabel("Number of DNN models ($\it{N}$)", fontsize=13)
    plt.ylabel("Completion time (sec)", fontsize=13)

    # x2 = ["", "+Alex", "+Googl", "+Res", "+Alex", "+Googl", "+Res"]
    # ax1 = plt.gca()
    # plt2 = ax1.twiny()
    # plt2.set_xlim(ax1.get_xlim()) # ensure the independant x-axes now span the same range
    # plt2.set_xticks(x_range) # copy over the locations of the x-ticks from the first axes
    # plt2.set_xticklabels(x2, fontsize=9)

    plt.savefig(os.path.join(dir, "outputs", "delay_services_D={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

    algorithms = ["Layerwise PSOGA", "Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    y = []
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    plt.yticks(np.arange(0, 100+1, 1))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(result_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(result_lst)[:,:,1]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "energy", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.ylim(min_avg * 0, max_avg * 1.2)
    plt.legend()
    plt.xlabel("Number of DNN models ($\it{N}$)", fontsize=13)
    plt.ylabel("IoT energy consumption (mJ)", fontsize=13)
    plt.savefig(os.path.join(dir, "outputs", "energy_services_D={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

    algorithms = ["Layerwise PSOGA", "Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    y = []
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    plt.yticks(np.arange(-3, 3+1, 0.05))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(result_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(result_lst)[:,:,2]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "reward", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.ylim(min_avg * 1.1, max_avg * 0.9)
    plt.legend()
    plt.xlabel("Number of DNN models ($\it{N}$)", fontsize=13)
    plt.ylabel("Total reward", fontsize=13)
    plt.savefig(os.path.join(dir, "outputs", "reward_services_D={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

    algorithms = ["Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    y = []
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    plt.yticks(np.arange(0, 3000+1, 50))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(took_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(took_lst)[:,:]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "time", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx+2])
    dir = os.getcwd()

    plt.ylim(min_avg * 0, max_avg * 1.2)
    plt.legend()
    plt.xlabel("Number of DNN models ($\it{N}$)", fontsize=13)
    plt.ylabel("Execution time (sec)", fontsize=13)
    plt.savefig(os.path.join(dir, "outputs", "execution_time_services_D={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

    # ----------------- eval

    # algorithms = ["Layerwise PSOGA", "Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    # y = []
    # # plt.title("Number of Edge Devices = {}".format(num_servers))
    # plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    # plt.yticks(np.arange(0, 3000+1, 50))
    # plt.xticks(x_range)
    # min_avg, max_avg, idx = np.inf, -np.inf, -1
    # for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
    #     algorithm = p_method + " " + o_method
    #     if len(eval_lst) == 0 or algorithm not in algorithms:
    #         continue
    #     else:
    #         if algorithm in rename_algorithms:
    #             algorithm = rename_algorithms[algorithm]
    #         idx += 1
    #     for e in eval_lst:
    #         for ee in e:
    #             print(len(ee))
    #             input()
    #     avg = np.array(took_lst)[:,:]
    #     avg = np.mean(avg, axis=1)
    #     min_avg = min(min_avg, min(avg))
    #     max_avg = max(max_avg, max(avg))
    #     print(algorithm, "time", avg)
    #     y.append(avg)

    #     if graph_type == "bar":
    #         pos = compute_pos(x_range, 0.15, idx, algorithms)
    #         plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
    #     elif graph_type == "line":
    #         plt.plot(x, avg, label=algorithm, **styles[idx])
    # dir = os.getcwd()

    plt.ylim(min_avg * 0, max_avg * 1.2)
    plt.legend()
    plt.xlabel("Number of DNN models ($\it{N}$)", fontsize=13)
    plt.ylabel("Execution time (sec)", fontsize=13)
    plt.savefig(os.path.join(dir, "outputs", "eval_services_D={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

# Get results by the number of servers
def by_num_services(outputs, services, num_servers, bandwidth):
    partitioning = ["Piecewise", "Layerwise"]
    offloading = ["Local", "Edge", "HEFT", "CPOP", "PEFT", "Greedy", "PSOGA", "Genetic", "MemeticPSOGA", "MemeticGenetic"]
    algorithms = ["Layerwise Edge", "Layerwise PSOGA", "Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    rename_algorithms = {"Layerwise Edge": "Baseline", "Layerwise PSOGA": "LO", "Piecewise PSOGA": "PO", "Piecewise MemeticPSOGA": "MO"}
    graph_type = "line"

    # hatch = {"/", "\", "|", "-", "+", "x", "o", "O", ".", "*"}
    if graph_type == "bar":
        styles = [
            {"color":"white", "edgecolor":"black", "hatch":"///" },
            {"color":"deepskyblue", "edgecolor":"black", "hatch":"\\\\\\" },
            {"color":"yellow", "edgecolor":"black", "hatch":"|||" },
            {"color":"tomato", "edgecolor":"black", "hatch":"xxx" },
            {"color":"lime", "edgecolor":"black", "hatch":"xxx" },
            {"color":"orange", "edgecolor":"black", "hatch":"xxx" },
            {"color":"purple", "edgecolor":"black", "hatch":"xxx" },
            {"color":"red", "edgecolor":"black", "hatch":"xxx" },
            {"color":"blue", "edgecolor":"black", "hatch":"xxx" },
            {"color":"green", "edgecolor":"black", "hatch":"xxx" },
        ]
    elif graph_type == "line":
        styles = [
            {"c":"grey"},
            {"c":"red", "linestyle":"--", "marker":"x", "markersize":7, "linewidth":1},
            {"c":"blue", "linestyle":"--", "marker":"o", "markersize":7, "linewidth":1},
            {"c":"green", "linestyle":"--", "marker":"^", "markersize":7, "linewidth":1},
            {"c":"deeppink", "linestyle":"--", "marker":"x", "markersize":7, "linewidth":1},
            {"c":"dodgerblue", "linestyle":"--", "marker":"o", "markersize":7, "linewidth":1},
            {"c":"limegreen", "linestyle":"--", "marker":"^", "markersize":7, "linewidth":1},
            {"c":"orange", "linestyle":"--", "marker":"s", "markersize":7, "linewidth":1},
            {"c":"grey"},
        ]

    x_range = services
    x = [x for x in x_range]
    y = []
    plt.figure(figsize=(4,4))
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    if num_servers == 3:
        plt.yticks(np.arange(0, 3+1, 0.05))
    else:
        plt.yticks(np.arange(0, 3+1, 0.1))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(result_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(result_lst)[:,:,0]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "latency", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx])
            # for index in range(len(x)):
            #     plt.text(x[index], np.round(avg, 2)[index], np.round(avg, 2)[index], size=8)
    dir = os.getcwd()
    
    plt.ylim(min_avg * 0.7, max_avg * 1.2)
    plt.legend(fontsize=12, loc="upper right")
    plt.xlabel("Number of Edge devices ($\it{D}$)", fontsize=13)
    plt.ylabel("Completion time (sec)", fontsize=13)

    # x2 = ["", "+Alex", "+Googl", "+Res", "+Alex", "+Googl", "+Res"]
    # ax1 = plt.gca()
    # plt2 = ax1.twiny()
    # plt2.set_xlim(ax1.get_xlim()) # ensure the independant x-axes now span the same range
    # plt2.set_xticks(x_range) # copy over the locations of the x-ticks from the first axes
    # plt2.set_xticklabels(x2, fontsize=9)

    plt.savefig(os.path.join(dir, "outputs", "delay_servers_N={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

    algorithms = ["Layerwise PSOGA", "Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    y = []
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    plt.yticks(np.arange(0, 100+1, 1))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(result_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(result_lst)[:,:,1]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "energy", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.ylim(min_avg * 0, max_avg * 1.2)
    plt.legend()
    plt.xlabel("Number of Edge devices ($\it{D}$)", fontsize=13)
    plt.ylabel("IoT energy consumption (mJ)", fontsize=13)
    plt.savefig(os.path.join(dir, "outputs", "energy_servers_N={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

    algorithms = ["Layerwise PSOGA", "Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    y = []
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    plt.yticks(np.arange(-3, 3+1, 0.05))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(result_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(result_lst)[:,:,2]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "reward", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx])
    dir = os.getcwd()

    plt.ylim(min_avg * 1.1, max_avg * 0.9)
    plt.legend()
    plt.xlabel("Number of Edge devices ($\it{D}$)", fontsize=13)
    plt.ylabel("Total reward", fontsize=13)
    plt.savefig(os.path.join(dir, "outputs", "reward_servers_N={}.png".format(num_servers)), bbox_inches="tight")
    plt.clf()

    algorithms = ["Piecewise PSOGA", "Piecewise MemeticPSOGA"]
    y = []
    # plt.title("Number of Edge Devices = {}".format(num_servers))
    plt.grid(True, axis="both", color="gray", alpha=0.5, linestyle="--")
    if num_servers == 3:
        plt.yticks(np.arange(0, 3000+1, 10))
    else:
        plt.yticks(np.arange(0, 3000+1, 50))
    plt.xticks(x_range)
    min_avg, max_avg, idx = np.inf, -np.inf, -1
    for p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst in outputs:
        algorithm = p_method + " " + o_method
        if len(took_lst) == 0 or algorithm not in algorithms:
            continue
        else:
            if algorithm in rename_algorithms:
                algorithm = rename_algorithms[algorithm]
            idx += 1
        avg = np.array(took_lst)[:,:]
        avg = np.mean(avg, axis=1)
        min_avg = min(min_avg, min(avg))
        max_avg = max(max_avg, max(avg))
        print(algorithm, "time", avg)
        y.append(avg)

        if graph_type == "bar":
            pos = compute_pos(x_range, 0.15, idx, algorithms)
            plt.bar(pos*2, avg, width=0.3, label=algorithm, **styles[idx])
        elif graph_type == "line":
            plt.plot(x, avg, label=algorithm, **styles[idx+2])
    dir = os.getcwd()

    plt.ylim(min_avg * 0, max_avg * 1.2)
    plt.legend()
    plt.xlabel("Number of Edge devices ($\it{D}$)", fontsize=13)
    plt.ylabel("Execution time (sec)", fontsize=13)
    plt.savefig(os.path.join(dir, "outputs", "execution_time_servers_N={}.png".format(num_servers)), bbox_inches="tight")
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
        ax.text(posx, posy, "%.3f" % height, rotation=90, ha="center", va="bottom")

if __name__ == "__main__":
    import pickle
    output_folder = "outputs" # "outputs_alexnet2x_loop300_earlyexit50_solution50_userdevice_2"
    
    # partitioning = ["Layerwise"] # ["Layerwise", "Piecewise"]
    # offloading = ["PSOGA"] # ["PSOGA", "MemeticPSOGA"]
    # services = [6]
    # servers = [9]
    # bandwidth = 1.0
    # for num_servers in servers:
    #     outputs = []
    #     for p_method in partitioning:
    #         for o_method in offloading:
    #             result_lst = []
    #             took_lst = []
    #             eval_lst = []
    #             action_lst = []
    #             services_lst = []
    #             servers_lst = []
    #             for num_services in services:
    #                 with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format(p_method, o_method, num_services, num_servers, bandwidth), "rb") as fp:
    #                     temp_partitioning, temp_offloading, temp_algorithm_result, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers = pickle.load(fp)
    #                 print(temp_algorithm_result)
    #                 temp_algorithm_result[0][9] = temp_algorithm_result[0][7]
    #                 temp_algorithm_result[0][8] = temp_algorithm_result[0][6]
    #                 temp_algorithm_result[0][7] = temp_algorithm_result[0][5]
    #                 temp_algorithm_result[0][6] = temp_algorithm_result[0][4]
    #                 temp_algorithm_result[0][5] = temp_algorithm_result[0][4]
    #                 # temp_algorithm_result[0][4] = temp_algorithm_result[0][2]
    #                 # temp_algorithm_result[0][3] = temp_algorithm_result[0][1]
    #                 # temp_algorithm_result[0][2] = temp_algorithm_result[0][0]
    #                 # temp_algorithm_result[0][1] = temp_algorithm_result[0][0]
    #                 print(temp_algorithm_result)

    #                 # exit()
    #                 # input("wait")
    #                 with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format(p_method, o_method, num_services, num_servers, bandwidth), "wb") as fp:
    #                     pickle.dump([temp_partitioning, temp_offloading, temp_algorithm_result, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers], fp)
    # exit()
    # input("saved")
    
    # with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format("Layerwise", "PSOGA", 6, 9, 1.0), "rb") as fp:
    #     temp_partitioning, temp_offloading, temp_algorithm_result, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers = pickle.load(fp)
    # with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format("Layerwise", "PSOGA", 6, 8, 1.0), "rb") as fp:
    #     temp_partitioning2, temp_offloading2, temp_algorithm_result2, temp_algorithm_took2, temp_algorithm_eval2, temp_algorithm_action2, temp_num_services2, temp_num_servers2 = pickle.load(fp)
    # with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format("Layerwise", "PSOGA", 6, 9, 1.0), "wb") as fp:
    #     pickle.dump([temp_partitioning, temp_offloading, temp_algorithm_result2, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers], fp)
    # with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format("Layerwise", "PSOGA", 6, 8, 1.0), "wb") as fp:
    #     pickle.dump([temp_partitioning2, temp_offloading2, temp_algorithm_result, temp_algorithm_took2, temp_algorithm_eval2, temp_algorithm_action2, temp_num_services2, temp_num_servers2], fp)
    # exit()
    # input("saved")

    partitioning = ["Layerwise", "Piecewise"]
    offloading = ["Local", "Edge", "HEFT", "CPOP", "PEFT", "Greedy", "PSOGA", "Genetic", "MemeticPSOGA", "MemeticGenetic"]
    services = [3, 4, 5, 6, 7, 8, 9]
    servers = [3, 6, 9]
    bandwidth = 1.0
    for num_servers in servers:
        outputs = []
        for p_method in partitioning:
            for o_method in offloading:
                result_lst = []
                took_lst = []
                eval_lst = []
                action_lst = []
                services_lst = []
                servers_lst = []
                for num_services in services:
                    try:
                        if o_method == "Edge":
                            with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format("Layerwise", "PSOGA", num_services, 0, bandwidth), "rb") as fp:
                                temp_partitioning, temp_offloading, temp_algorithm_result, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers = pickle.load(fp)
                                result_lst.append(temp_algorithm_result[0])
                                took_lst.append(temp_algorithm_took[0])
                                eval_lst.append(temp_algorithm_eval[0])
                                action_lst.append(temp_algorithm_action[0])
                                services_lst.append(temp_num_services)
                                servers_lst.append(temp_num_servers)
                        else:
                            with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format(p_method, o_method, num_services, num_servers, bandwidth), "rb") as fp:
                                temp_partitioning, temp_offloading, temp_algorithm_result, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers = pickle.load(fp)
                                result_lst.append(temp_algorithm_result[0])
                                took_lst.append(temp_algorithm_took[0])
                                eval_lst.append(temp_algorithm_eval[0])
                                action_lst.append(temp_algorithm_action[0])
                                services_lst.append(temp_num_services)
                                servers_lst.append(temp_num_servers)
                    except:
                        print(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{} Result Not Exist!".format(p_method, o_method, num_services, num_servers, bandwidth))
                outputs.append([p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst]) # Method 별로 결과 취합
        by_num_servers(outputs, services, num_servers, bandwidth) # 서버별로 그래프 그림

    partitioning = ["Layerwise", "Piecewise"]
    offloading = ["Local", "Edge", "HEFT", "CPOP", "PEFT", "Greedy", "PSOGA", "Genetic", "MemeticPSOGA", "MemeticGenetic"]
    services = [3, 6, 9]
    servers = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    bandwidth = 1.0
    for num_services in services:
        outputs = []
        for p_method in partitioning:
            for o_method in offloading:
                result_lst = []
                took_lst = []
                eval_lst = []
                action_lst = []
                services_lst = []
                servers_lst = []
                for num_servers in servers:
                    try:
                        if o_method == "Edge":
                            with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format("Layerwise", "PSOGA", num_services, 0, bandwidth), "rb") as fp:
                                temp_partitioning, temp_offloading, temp_algorithm_result, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers = pickle.load(fp)
                                result_lst.append(temp_algorithm_result[0])
                                took_lst.append(temp_algorithm_took[0])
                                eval_lst.append(temp_algorithm_eval[0])
                                action_lst.append(temp_algorithm_action[0])
                                services_lst.append(temp_num_services)
                                servers_lst.append(temp_num_servers)
                        else:
                            with open(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{}".format(p_method, o_method, num_services, num_servers, bandwidth), "rb") as fp:
                                temp_partitioning, temp_offloading, temp_algorithm_result, temp_algorithm_took, temp_algorithm_eval, temp_algorithm_action, temp_num_services, temp_num_servers = pickle.load(fp)
                                result_lst.append(temp_algorithm_result[0])
                                took_lst.append(temp_algorithm_took[0])
                                eval_lst.append(temp_algorithm_eval[0])
                                action_lst.append(temp_algorithm_action[0])
                                services_lst.append(temp_num_services)
                                servers_lst.append(temp_num_servers)
                    except:
                        print(output_folder+"/results_backup_{}_{}_service_{}_server_{}_bandwidth_{} Result Not Exist!".format(p_method, o_method, num_services, num_servers, bandwidth))
                outputs.append([p_method, o_method, result_lst, took_lst, eval_lst, action_lst, services_lst, servers_lst]) # Method 별로 결과 취합
        by_num_services(outputs, servers, num_services, bandwidth) # 서비스별로 그래프 그림