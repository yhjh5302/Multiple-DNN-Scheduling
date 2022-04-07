from copy import deepcopy
from re import S
import numpy as np
import multiprocessing as mp
import math
import time

class MultiLevelGraphPartitioning:
    def __init__(self, dataset, num_partitions):
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_node = dataset.num_partitions
        self.num_partitions = num_partitions

        self.average_computing_power = dataset.system_manager.average_computing_power
        self.average_bandwidth = dataset.system_manager.average_bandwidth

        self.proc_time = np.zeros(self.num_node)
        self.tran_time = dict()
        # init times
        for c in self.system_manager.service_set.partitions:
            if len(c.predecessors) == 0:
                self.get_time(c)

        self.rank_up = np.zeros(self.num_node)
        self.rank_down = np.zeros(self.num_node)
        # init ranks
        for id in range(self.num_node):
            if self.rank_up[id] == 0:
                self.calc_rank_up(id)
        for id in range(self.num_node)[::-1]:
            if self.rank_down[id] == 0:
                self.calc_rank_down(id)

        self.path_e = dict()
        # init path_e
        self.calc_path_e()

    def get_time(self, partition):
        if self.proc_time[partition.id] == 0:
            self.proc_time[partition.id] = partition.workload_size / self.average_computing_power
        for succ in partition.successors:
            if (partition.id, succ.id) not in self.tran_time:
                self.tran_time[(partition.id, succ.id)] = partition.get_output_data_size(succ) / self.average_bandwidth
                self.get_time(succ)

    def calc_rank_up(self, id):    # rank up for multi-level partitioning
        lst = [0,]
        for (pred_id, succ_id) in self.tran_time.keys():
            if pred_id == id:
                if self.rank_up[succ_id] == 0:
                    self.calc_rank_up(succ_id)
                lst.append(self.proc_time[pred_id] + self.tran_time[(pred_id, succ_id)] + self.rank_up[succ_id])
        self.rank_up[id] = max(lst)

    def calc_rank_down(self, id):    # rank down for multi-level partitioning
        lst = [self.proc_time[id],]
        for (pred_id, succ_id) in self.tran_time.keys():
            if succ_id == id:
                if self.rank_down[pred_id] == 0:
                    self.calc_rank_down(pred_id)
                lst.append(self.proc_time[succ_id] + self.tran_time[(pred_id, succ_id)] + self.rank_down[pred_id])
        self.rank_down[id] = max(lst)

    def calc_path_e(self):
        for (pred_id, succ_id) in self.tran_time.keys():
            self.path_e[(pred_id, succ_id)] = self.rank_down[pred_id] + self.proc_time[succ_id] + self.tran_time[(pred_id, succ_id)] + self.rank_up[succ_id]

    def refinement(self, graph, W_e, next_graph, next_W_e):
        coarsened_graph = np.unique(graph)
        next_coarsened_graph = np.unique(next_graph)
        boundary_node_set = []
        gain_queue = []
        # find boundary node set and calculate initial gain priority queue
        for node in next_coarsened_graph:
            if next((edge for edge in next_W_e.keys() if node in edge and graph[edge[0]] != graph[edge[1]]), None):
                edges = [edge for edge in next_W_e.keys() if node in edge]
                uncut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
                cut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
                boundary_node_set.append(node)
                gain_queue.append(cut_edges_gain - uncut_edges_gain)
        
        gain_priority_queue = sorted(zip(gain_queue, boundary_node_set), reverse=True)
        print("boundary_node_set", len(boundary_node_set), boundary_node_set)
        print(gain_priority_queue)

        # find 
        sequence = []
        gain_sum = 0
        for (gain, node) in gain_priority_queue:
            gain_sum += gain
            sequence.append((node, gain_sum))
        
        sequence.sort(key=lambda x: x[1], reverse=True)
            
        for (node, gain_sum) in sequence:
            edges = [edge for edge in next_W_e.keys() if node in edge]
            neighbor_partitions = [graph[edge[1]] if node == edge[0] else graph[edge[0]] for edge in edges]
            uncut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
            cut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
            gain = cut_edges_gain - uncut_edges_gain
            testt = graph[node]
            print(self.system_manager.service_set.partitions[node].layer_name, [self.system_manager.service_set.partitions[u].layer_name for u in np.where(graph == testt)[0]], "gain", gain)
            for neighbor in np.unique(neighbor_partitions):
                temp = graph[node]
                graph[np.where(next_graph == next_graph[node])] = neighbor
                uncut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
                cut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
                new_gain = cut_edges_gain - uncut_edges_gain
                if gain > new_gain:
                    gain = new_gain
                else:
                    graph[np.where(next_graph == next_graph[node])] = temp
            print(self.system_manager.service_set.partitions[node].layer_name, [self.system_manager.service_set.partitions[u].layer_name for u in np.where(graph == testt)[0]], "gain", gain)
            input()

            gain_priority_queue.sort(key=lambda x: x[0])
        print(sequence)
        input()
        return graph, W_e

    def run_algo(self):
        from copy import deepcopy
        from itertools import groupby
        graph = np.arange(start=0, stop=self.num_node)
        W_e = deepcopy(self.path_e)
        tran_time = deepcopy(self.tran_time)
        rank_down = deepcopy(self.rank_down)
        cut_graph = [(deepcopy(graph), deepcopy(W_e))]
        theta = self.num_partitions
        print(len(W_e), len(np.unique(graph)))

        # Latency-path coarsening
        while len(np.unique(graph)) >= theta:
            node_order = np.array(sorted(zip(rank_down, graph), reverse=False), dtype=np.int32)[:,1]
            coarsened = np.full(shape=self.num_node, fill_value=False, dtype=np.bool8)
            for u in node_order:
                if coarsened[u] == False:
                    coarsened[np.where(graph==u)] = True
                    max_W_e = 0
                    max_v = None

                    # find max W_e
                    for v in np.unique(graph):
                        if coarsened[v] == True:
                            continue
                        if (u, v) in W_e and W_e[(u, v)] > max_W_e:
                            max_W_e = W_e[(u, v)]
                            max_v = v
                        elif (v, u) in W_e and W_e[(v, u)] > max_W_e:
                            max_W_e = W_e[(v, u)]
                            max_v = v
                    if max_v == None:
                        for v in np.unique(graph):
                            if (u, v) in W_e and W_e[(u, v)] > max_W_e:
                                max_W_e = W_e[(u, v)]
                                max_v = v
                            elif (v, u) in W_e and W_e[(v, u)] > max_W_e:
                                max_W_e = W_e[(v, u)]
                                max_v = v

                    # coarsening
                    coarsened[np.where(graph==max_v)] = True
                    graph[np.where(graph==max_v)] = u

                    # update rank_down
                    rank_down[np.where(graph==u)] = max(rank_down[u], rank_down[max_v])

                    # update W_e
                    delete_lst = []
                    W_e_append_lst = dict()
                    tran_time_append_lst = dict()
                    for (pred_id, succ_id) in W_e:
                        if pred_id == max_v:
                            if succ_id == u:
                                pass
                            elif (u, succ_id) in W_e:
                                W_e[(u, succ_id)] = W_e[(u, succ_id)] + W_e[(max_v, succ_id)]
                                tran_time[(u, succ_id)] = tran_time[(u, succ_id)] + tran_time[(max_v, succ_id)]
                            else:
                                W_e_append_lst[(u, succ_id)] = W_e[(max_v, succ_id)]
                                tran_time_append_lst[(u, succ_id)] = tran_time[(max_v, succ_id)]
                            delete_lst.append((max_v, succ_id))
                        elif succ_id == max_v:
                            if pred_id == u:
                                pass
                            elif (pred_id, u) in W_e:
                                W_e[(pred_id, u)] = W_e[(pred_id, u)] + W_e[(pred_id, max_v)]
                                tran_time[(pred_id, u)] = tran_time[(pred_id, u)] + tran_time[(pred_id, max_v)]
                            else:
                                W_e_append_lst[(pred_id, u)] = W_e[(pred_id, max_v)]
                                tran_time_append_lst[(pred_id, u)] = tran_time[(pred_id, max_v)]
                            delete_lst.append((pred_id, max_v))
                    W_e.update(W_e_append_lst)
                    tran_time.update(tran_time_append_lst)
                    for key in delete_lst:
                        del W_e[key]
                        del tran_time[key]
                    if not len(np.unique(graph)) >= theta:
                        break

            if len(np.unique(graph)) >= theta:
                cut_graph.append((deepcopy(graph), deepcopy(W_e)))

                #### testout
                print(len(W_e), len(np.unique(graph)))
                # print(coarsened, graph)
                # lst = [(graph[id], self.system_manager.service_set.partitions[id].layer_name) for id in range(self.num_node)]
                # print([[item[1] for item in items] for key, items in groupby(sorted(lst, key=lambda x: x[0]), lambda x: x[0])])
                # input()
        
        # Latency-path initial partition
        graph, W_e = cut_graph.pop()
        while len(cut_graph) > 0:
            graph, W_e = self.refinement(graph, W_e, *cut_graph.pop())
        

        input("Latency-path coarsening done...")
        return 