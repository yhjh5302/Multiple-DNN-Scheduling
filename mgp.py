from copy import deepcopy
from re import S
import numpy as np
import multiprocessing as mp
import math
import time
from itertools import groupby

class MultilevelGraphPartitioning:
    def __init__(self, dataset, k):
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_partitions = dataset.num_partitions
        self.k = k

        self.coarsened_graph = dataset.coarsened_graph
        self.average_computing_power = dataset.system_manager.average_computing_power
        self.average_bandwidth = dataset.system_manager.average_bandwidth
        self.rank = self.system_manager.rank_u

    def get_time(self, partition):
        if self.proc_time[partition.id] == 0:
            self.proc_time[partition.id] = partition.workload_size / self.average_computing_power
        for succ in partition.successors:
            if (partition.id, succ.id) not in self.tran_time:
                self.tran_time[(partition.id, succ.id)] = partition.get_output_data_size(succ) / self.average_bandwidth
                self.get_time(succ)

    def refinement(self, graph, tran_time, next_graph, next_tran_time, svc_id):
        print(len(np.unique(next_graph)))
        coarsened_graph = np.unique(graph)
        next_coarsened_graph = np.unique(next_graph)
        boundary_node_set = []
        gain_queue = []
        # find boundary node set and calculate initial gain priority queue
        for node in next_coarsened_graph:
            if next((edge for edge in next_tran_time.keys() if node in edge and graph[edge[0]] != graph[edge[1]]), None):
                edges = [edge for edge in next_tran_time.keys() if node in edge]
                uncut_edges_gain = sum([next_tran_time[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
                cut_edges_gain = sum([next_tran_time[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
                boundary_node_set.append(node)
                gain_queue.append(uncut_edges_gain - cut_edges_gain)
        
        gain_priority_queue = sorted(zip(gain_queue, boundary_node_set), reverse=True)
        print("gain_priority_queue", len(gain_priority_queue), gain_priority_queue)
        input()

        # find 
        sequence = []
        gain_sum = 0
        for (gain, node) in gain_priority_queue:
            gain_sum += gain
            sequence.append((node, gain_sum))
        
        sequence.sort(key=lambda x: x[1], reverse=True)
            
        for (node, gain_sum) in sequence:
            edges = [edge for edge in next_tran_time.keys() if node in edge]
            neighbor_partitions = [graph[edge[1]] if node in edge else graph[edge[0]] for edge in edges]
            uncut_edges_gain = sum([next_tran_time[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
            cut_edges_gain = sum([next_tran_time[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
            gain = uncut_edges_gain - cut_edges_gain
            testt = graph[node]
            #print(self.system_manager.service_set.partitions[node].layer_name, [self.system_manager.service_set.partitions[u].layer_name for u in np.where(graph == testt)[0]], "gain", gain)
            for neighbor in np.unique(neighbor_partitions):
                temp = graph[node]
                graph[np.where(next_graph == next_graph[node])] = neighbor
                uncut_edges_gain = sum([next_tran_time[edge] for edge in edges if graph[edge[0]] == graph[edge[1]]])
                cut_edges_gain = sum([next_tran_time[edge] for edge in edges if graph[edge[0]] != graph[edge[1]]])
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
        print(len(np.unique(graph)))
        # print(coarsened, graph)
        lst = [(graph[id], self.system_manager.service_set.services[svc_id].partitions[id].layer_name) for id in range(graph_size)]
        print([[item[1] for item in items] for key, items in groupby(sorted(lst, key=lambda x: x[0]), lambda x: x[0])])
        input()
        return graph, tran_time

    def run_algo(self):
        new_coarsened_graph = []
        for svc_id, graph in enumerate(self.coarsened_graph):
            graph_size = len(graph)
            graph = np.arange(graph_size)
            theta = math.floor(self.k * 2)
            partition_piece_map = np.array([c.piece_idx for c in self.system_manager.service_set.services[svc_id].partitions])
            # print(svc_id, graph_size)

            # init times
            self.proc_time = np.zeros(graph_size)
            self.tran_time = dict()
            for c in self.system_manager.service_set.services[svc_id].partitions:
                if len(c.predecessors) == 0:
                    self.get_time(c)

            tran_time = deepcopy(self.tran_time)
            rank = deepcopy(self.rank[graph])
            cut_graph = [(deepcopy(graph), deepcopy(tran_time))]

            # Latency-path coarsening
            # print(len(np.unique(graph)), theta)
            while len(np.unique(graph)) >= theta:
                node_order = np.array(sorted(zip(rank, graph), reverse=True), dtype=np.int32)[:,1]
                coarsened = np.full(shape=graph_size, fill_value=False, dtype=np.bool8)
                for i in node_order:
                    if coarsened[i] == False:
                        coarsened[np.where(graph==i)] = True
                        max_T_tr_ij = 0
                        max_j = None

                        # find max T_ij
                        for j in np.unique(graph):
                            if partition_piece_map[i] == partition_piece_map[j] and (i, j) in tran_time and tran_time[(i, j)] > max_T_tr_ij:
                                max_T_tr_ij = tran_time[(i, j)]
                                max_j = j
                        if max_j == None:
                            continue

                        # coarsening
                        coarsened[np.where(graph==max_j)] = True
                        graph[np.where(graph==max_j)] = i

                        # update rank
                        rank[np.where(graph==i)] = max(rank[i], rank[max_j])

                        # update T_tr_ik
                        delete_lst = []
                        append_lst = dict()
                        for (pred_id, succ_id) in tran_time:
                            if pred_id == max_j:
                                if succ_id == i:
                                    pass
                                elif (i, succ_id) in tran_time:
                                    tran_time[(i, succ_id)] = tran_time[(i, succ_id)] + tran_time[(max_j, succ_id)]
                                else:
                                    append_lst[(i, succ_id)] = tran_time[(max_j, succ_id)]
                                delete_lst.append((max_j, succ_id))
                            elif succ_id == max_j:
                                if pred_id == i:
                                    pass
                                elif (pred_id, i) in tran_time:
                                    tran_time[(pred_id, i)] = tran_time[(pred_id, i)] + tran_time[(pred_id, max_j)]
                                else:
                                    append_lst[(pred_id, i)] = tran_time[(pred_id, max_j)]
                                delete_lst.append((pred_id, max_j))
                        tran_time.update(append_lst)
                        for key in delete_lst:
                            del tran_time[key]
                        
                        # node name ordering (for readable result)
                        if i > max_j:
                            graph[np.where(graph==i)] = max_j
                            delete_lst = []
                            append_lst = dict()
                            for old_key in tran_time.keys():
                                if i in old_key:
                                    delete_lst.append(old_key)
                                    if old_key[0] == i:
                                        new_key = (max_j, old_key[1])
                                    elif old_key[1] == i:
                                        new_key = (old_key[0], max_j)
                                    append_lst[new_key] = tran_time[old_key]
                            tran_time.update(append_lst)
                            for key in delete_lst:
                                del tran_time[key]
                        tran_time = {k: v for k, v in sorted(tran_time.items(), key = lambda item: item[0])}

                        if not len(np.unique(graph)) >= theta:
                            break

                if len(np.unique(graph)) >= theta:
                    cut_graph.append((deepcopy(graph), deepcopy(tran_time)))

                #### testout
                # print(len(np.unique(graph)))
                # # print(coarsened, graph)
                # lst = [(graph[id], self.system_manager.service_set.services[svc_id].partitions[id].layer_name) for id in range(graph_size)]
                # print([[item[1] for item in items] for key, items in groupby(sorted(lst, key=lambda x: x[0]), lambda x: x[0])])
                # input()
        
            # Latency-path initial partition
            graph, tran_time = deepcopy(cut_graph[-1])
            while len(np.unique(graph)) > self.k:
                node_order = np.array(sorted(zip(rank, graph), reverse=True), dtype=np.int32)[:,1]
                coarsened = np.full(shape=graph_size, fill_value=False, dtype=np.bool8)
                for i in node_order:
                    if coarsened[i] == False:
                        coarsened[np.where(graph==i)] = True
                        max_T_tr_ij = 0
                        max_j = None

                        # find max tran_time
                        for j in np.unique(graph):
                            if partition_piece_map[i] == partition_piece_map[j] and (i, j) in tran_time and tran_time[(i, j)] > max_T_tr_ij:
                                max_T_tr_ij = tran_time[(i, j)]
                                max_j = j
                        if max_j == None:
                            continue

                        # coarsening
                        coarsened[np.where(graph==max_j)] = True
                        graph[np.where(graph==max_j)] = i

                        # update rank
                        rank[np.where(graph==i)] = max(rank[i], rank[max_j])

                        # update T_tr_ik
                        delete_lst = []
                        append_lst = dict()
                        for (pred_id, succ_id) in tran_time:
                            if pred_id == max_j:
                                if succ_id == i:
                                    pass
                                elif (i, succ_id) in tran_time:
                                    tran_time[(i, succ_id)] = tran_time[(i, succ_id)] + tran_time[(max_j, succ_id)]
                                else:
                                    append_lst[(i, succ_id)] = tran_time[(max_j, succ_id)]
                                delete_lst.append((max_j, succ_id))
                            elif succ_id == max_j:
                                if pred_id == i:
                                    pass
                                elif (pred_id, i) in tran_time:
                                    tran_time[(pred_id, i)] = tran_time[(pred_id, i)] + tran_time[(pred_id, max_j)]
                                else:
                                    append_lst[(pred_id, i)] = tran_time[(pred_id, max_j)]
                                delete_lst.append((pred_id, max_j))
                        tran_time.update(append_lst)
                        for key in delete_lst:
                            del tran_time[key]
                        
                        # node name ordering (for readable result)
                        if i > max_j:
                            graph[np.where(graph==i)] = max_j
                            delete_lst = []
                            append_lst = dict()
                            for old_key in tran_time.keys():
                                if i in old_key:
                                    delete_lst.append(old_key)
                                    if old_key[0] == i:
                                        new_key = (max_j, old_key[1])
                                    elif old_key[1] == i:
                                        new_key = (old_key[0], max_j)
                                    append_lst[new_key] = tran_time[old_key]
                            tran_time.update(append_lst)
                            for key in delete_lst:
                                del tran_time[key]
                        tran_time = {k: v for k, v in sorted(tran_time.items(), key = lambda item: item[0])}
                    
                    if not len(np.unique(graph)) > self.k:
                        break

            ### testout
            # print(len(np.unique(graph)))
            # # print(coarsened, graph)
            # lst = [(graph[id], self.system_manager.service_set.services[svc_id].partitions[id].layer_name) for id in range(graph_size)]
            # print([[item[1] for item in items] for key, items in groupby(sorted(lst, key=lambda x: x[0]), lambda x: x[0])])
            # input()
            print(len(np.unique(graph)), theta)
            new_coarsened_graph.append(graph)
        return new_coarsened_graph

            # let's refine
        #     S_m = cut_graph.pop()
        #     for cg in reversed(cut_graph):
        #         next_graph, next_tran_time = cg
        #         graph, tran_time = self.refinement(graph, tran_time, next_graph, next_tran_time, svc_id
        #     new_coarsened_graph.append(graph)
        # return new_coarsened_graph