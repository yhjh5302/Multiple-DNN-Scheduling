from copy import deepcopy
from re import S
import numpy as np
import multiprocessing as mp
import math
import time
from itertools import groupby

class MultiLevelGraphPartitioning:
    def __init__(self, dataset):
        self.system_manager = dataset.system_manager

        self.average_computing_power = dataset.system_manager.average_computing_power
        self.average_bandwidth = dataset.system_manager.average_bandwidth

        self.proc_time = dict()
        self.tran_time = dict()
        self.rank_up = dict()
        self.rank_down = dict()
        self.path_e = dict()
        for svc in dataset.svc_set.services:
            self.proc_time[svc.id] = dict()
            self.tran_time[svc.id] = dict()
            # init times
            for c in svc.partitions:
                if len(c.predecessors) == 0:
                    self.get_time(svc.id, c)

            self.rank_up[svc.id] = dict()
            self.rank_down[svc.id] = dict()
            # init ranks
            for partition in svc.partitions:
                if partition.id not in self.rank_up[svc.id]:
                    self.calc_rank_up(svc.id, partition.id)
            for partition in reversed(svc.partitions):
                if partition.id not in self.rank_down[svc.id]:
                    self.calc_rank_down(svc.id, partition.id)

            self.path_e[svc.id] = dict()
            # init path_e
            self.calc_path_e(svc.id)

    def get_time(self, svc_id, partition):
        if partition.id not in self.proc_time[svc_id]:
            self.proc_time[svc_id][partition.id] = partition.workload_size / self.average_computing_power
        for succ in partition.successors:
            if (partition.id, succ.id) not in self.tran_time[svc_id]:
                self.tran_time[svc_id][(partition.id, succ.id)] = partition.get_output_data_size(succ) / self.average_bandwidth
                self.get_time(svc_id, succ)

    def calc_rank_up(self, svc_id, id):    # rank up for multi-level partitioning
        lst = [0,]
        for (pred_id, succ_id) in self.tran_time[svc_id].keys():
            if pred_id == id:
                if succ_id not in self.rank_up[svc_id]:
                    self.calc_rank_up(svc_id, succ_id)
                lst.append(self.proc_time[svc_id][pred_id] + self.tran_time[svc_id][(pred_id, succ_id)] + self.rank_up[svc_id][succ_id])
        self.rank_up[svc_id][id] = max(lst)

    def calc_rank_down(self, svc_id, id):    # rank down for multi-level partitioning
        lst = [self.proc_time[svc_id][id],]
        for (pred_id, succ_id) in self.tran_time[svc_id].keys():
            if succ_id == id:
                if pred_id not in self.rank_down[svc_id]:
                    self.calc_rank_down(svc_id, pred_id)
                lst.append(self.proc_time[svc_id][succ_id] + self.tran_time[svc_id][(pred_id, succ_id)] + self.rank_down[svc_id][pred_id])
        self.rank_down[svc_id][id] = max(lst)

    def calc_path_e(self, svc_id):
        for (pred_id, succ_id) in self.tran_time[svc_id].keys():
            self.path_e[svc_id][(pred_id, succ_id)] = self.rank_down[svc_id][pred_id] + self.proc_time[svc_id][succ_id] + self.tran_time[svc_id][(pred_id, succ_id)] + self.rank_up[svc_id][succ_id]

    def refinement(self, graph, W_e, tran_time, rank_down,
                   next_graph, next_W_e, next_tran_time, next_rank_down):
        print(len(np.unique(next_graph)))
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
                gain_queue.append(uncut_edges_gain - cut_edges_gain)
        
        gain_priority_queue = sorted(zip(gain_queue, boundary_node_set), reverse=True)
        # print("gain_priority_queue", len(gain_priority_queue), gain_priority_queue)

        # find 
        sequence = []
        gain_sum = 0
        for (gain, node) in gain_priority_queue:
            gain_sum += gain
            sequence.append((node, gain_sum))
        
        sequence.sort(key=lambda x: x[1], reverse=True)
            
        for (node, gain_sum) in sequence:
            edges = [edge for edge in next_W_e.keys() if node in edge]
            neighbor_partitions = [graph[edge[1]] if node in edge else graph[edge[0]] for edge in edges]
            uncut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
            cut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
            gain = uncut_edges_gain - cut_edges_gain
            testt = graph[node]
            #print(self.system_manager.service_set.partitions[node].layer_name, [self.system_manager.service_set.partitions[u].layer_name for u in np.where(graph == testt)[0]], "gain", gain)
            for neighbor in np.unique(neighbor_partitions):
                temp = graph[node]
                graph[np.where(next_graph == next_graph[node])] = neighbor
                uncut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
                cut_edges_gain = sum([next_W_e[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
                new_gain = uncut_edges_gain - cut_edges_gain
                if gain > new_gain:
                    gain = new_gain
                else:
                    graph[np.where(next_graph == next_graph[node])] = temp
            #print(self.system_manager.service_set.partitions[node].layer_name, [self.system_manager.service_set.partitions[u].layer_name for u in np.where(graph == testt)[0]], "gain", gain)
            #print([self.system_manager.service_set.partitions[u].layer_name for u in np.where(graph == graph[node])[0]])
            #input()

            gain_priority_queue.sort(key=lambda x: x[0])
        #### testout
        print(len(W_e), len(np.unique(graph)))
        # print(coarsened, graph)
        # lst = [(graph[p.id-p_start], self.system_manager.service_set.partitions[p.id].layer_name) for p in svc.partitions]
        # print([[item[1] for item in items] for key, items in groupby(sorted(lst, key=lambda x: x[0]), lambda x: x[0])])
        # input()
        return graph, W_e, tran_time, rank_down

    def run_algo(self, svc, num_partitions):
        graph = np.array([partition.id for partition in svc.partitions])
        rank_down = np.array([self.rank_down[svc.id][partition.id] for partition in svc.partitions])
        tran_time = deepcopy(self.tran_time[svc.id])
        W_e = deepcopy(self.path_e[svc.id])
        cut_graph = [(deepcopy(graph), deepcopy(W_e), deepcopy(tran_time), deepcopy(rank_down))]
        theta = math.floor(num_partitions * 4)
        p_start = min(graph)
        print(len(W_e), len(np.unique(graph)))

        # Latency-path coarsening
        while len(np.unique(graph)) >= theta:
            ordered_graph = np.array(sorted(zip(rank_down, graph), reverse=False), dtype=np.int32)[:,1]
            coarsened = np.full(shape=svc.num_partitions, fill_value=False, dtype=np.bool8)
            for u in ordered_graph:
                if coarsened[u-p_start] == False:
                    coarsened[np.where(graph==u)] = True
                    max_W_e = 0
                    max_v = None

                    # find max W_e
                    for v in np.unique(graph):
                        if (u, v) in W_e and W_e[(u, v)] > max_W_e:
                            max_W_e = W_e[(u, v)]
                            max_v = v
                        elif (u, v) in W_e and W_e[(u, v)] == max_W_e and coarsened[max_v-p_start] == True and coarsened[v-p_start] == False:
                            max_W_e = W_e[(u, v)]
                            max_v = v
                    if max_v == None:
                        for v in np.unique(graph):
                            if (v, u) in W_e and W_e[(v, u)] > max_W_e:
                                max_W_e = W_e[(v, u)]
                                max_v = v
                            elif (v, u) in W_e and W_e[(v, u)] == max_W_e and coarsened[max_v-p_start] == True and coarsened[v-p_start] == False:
                                max_W_e = W_e[(v, u)]
                                max_v = v

                    # coarsening
                    coarsened[np.where(graph==max_v)] = True
                    graph[np.where(graph==max_v)] = u

                    # update rank_down
                    rank_down[np.where(graph==u)] = max(rank_down[u-p_start], rank_down[max_v-p_start])

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
                    
                    # node name ordering (for readable result)
                    if u > max_v:
                        graph[np.where(graph==u)] = max_v
                        delete_lst = []
                        W_e_append_lst = dict()
                        tran_time_append_lst = dict()
                        for old_key in W_e.keys():
                            if u in old_key:
                                delete_lst.append(old_key)
                                if old_key[0] == u:
                                    new_key = (max_v, old_key[1])
                                elif old_key[1] == u:
                                    new_key = (old_key[0], max_v)
                                W_e_append_lst[new_key] = W_e[old_key]
                                tran_time_append_lst[new_key] = tran_time[old_key]
                        W_e.update(W_e_append_lst)
                        tran_time.update(tran_time_append_lst)
                        for key in delete_lst:
                            del W_e[key]
                            del tran_time[key]
                    W_e = {k: v for k, v in sorted(W_e.items(), key = lambda item: item[0])}
                    tran_time = {k: v for k, v in sorted(tran_time.items(), key = lambda item: item[0])}

                    if not len(np.unique(graph)) >= theta:
                        break

            if len(np.unique(graph)) >= theta:
                cut_graph.append((deepcopy(graph), deepcopy(W_e), deepcopy(tran_time), deepcopy(rank_down)))

                #### testout
                print(len(W_e), len(np.unique(graph)))
                # print(coarsened, graph)
                # lst = [(graph[p.id-p_start], self.system_manager.service_set.partitions[p.id].layer_name) for p in svc.partitions]
                # print([[item[1] for item in items] for key, items in groupby(sorted(lst, key=lambda x: x[0]), lambda x: x[0])])
                # input()
        
        # Latency-path initial partition
        graph, W_e, tran_time, rank_down = deepcopy(cut_graph[-1])
        temp_graph = []
        temp_graph.append((deepcopy(graph), deepcopy(W_e), deepcopy(tran_time), deepcopy(rank_down)))
        while len(np.unique(graph)) > num_partitions:
            ordered_graph = np.array(sorted(zip(rank_down, graph), reverse=False), dtype=np.int32)[:,1]
            coarsened = np.full(shape=svc.num_partitions, fill_value=False, dtype=np.bool8)
            for u in ordered_graph:
                if coarsened[u-p_start] == False:
                    coarsened[np.where(graph==u)] = True
                    max_W_e = 0
                    max_v = None

                    # find max W_e
                    for v in np.unique(graph):
                        if (u, v) in W_e and W_e[(u, v)] > max_W_e and coarsened[v-p_start] == False:
                            max_W_e = W_e[(u, v)]
                            max_v = v
                    if max_v == None:
                        continue

                    # coarsening
                    coarsened[np.where(graph==max_v)] = True
                    graph[np.where(graph==max_v)] = u

                    # update rank_down
                    rank_down[np.where(graph==u)] = max(rank_down[u-p_start], rank_down[max_v-p_start])

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
                    
                    # node name ordering (for readable result)
                    if u > max_v:
                        graph[np.where(graph==u)] = max_v
                        delete_lst = []
                        W_e_append_lst = dict()
                        tran_time_append_lst = dict()
                        for old_key in W_e.keys():
                            if u in old_key:
                                delete_lst.append(old_key)
                                if old_key[0] == u:
                                    new_key = (max_v, old_key[1])
                                elif old_key[1] == u:
                                    new_key = (old_key[0], max_v)
                                W_e_append_lst[new_key] = W_e[old_key]
                                tran_time_append_lst[new_key] = tran_time[old_key]
                        W_e.update(W_e_append_lst)
                        tran_time.update(tran_time_append_lst)
                        for key in delete_lst:
                            del W_e[key]
                            del tran_time[key]
                    W_e = {k: v for k, v in sorted(W_e.items(), key = lambda item: item[0])}
                    tran_time = {k: v for k, v in sorted(tran_time.items(), key = lambda item: item[0])}
                    
                    if not len(np.unique(graph)) >= num_partitions:
                        break
            if len(np.unique(graph)) >= num_partitions:
                temp_graph.append((deepcopy(graph), deepcopy(W_e), deepcopy(tran_time), deepcopy(rank_down)))
        graph, W_e, tran_time, rank_down = temp_graph.pop()

        ### testout
        print(len(W_e), len(np.unique(graph)))
        # print(coarsened, graph)
        # lst = [(graph[p.id-p_start], self.system_manager.service_set.partitions[p.id].layer_name) for p in svc.partitions]
        # print([[item[1] for item in items] for key, items in groupby(sorted(lst, key=lambda x: x[0]), lambda x: x[0])])
        # input()
        return graph, tran_time, self.proc_time