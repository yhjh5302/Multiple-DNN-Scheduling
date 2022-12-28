import numpy as np
import time
from copy import deepcopy


class Local:
    def __init__(self, dataset):
        self.dataset = dataset

    def run_algo(self):
        timer = time.time()
        x = np.array([[self.dataset.system_manager.net_manager.request_device[p.service.id] for p in self.dataset.svc_set.partitions] for t in range(self.dataset.num_timeslots)])
        self.dataset.system_manager.set_env(deployed_server=x[0])
        return (x, [np.max(self.dataset.system_manager.total_time_dp())], time.time() - timer)


class Edge:
    def __init__(self, dataset):
        self.dataset = dataset

    def run_algo(self):
        timer = time.time()
        x = np.full(shape=(self.dataset.num_timeslots, self.dataset.num_partitions), fill_value=list(self.dataset.system_manager.edge.keys())[0], dtype=np.int32)
        self.dataset.system_manager.set_env(deployed_server=x[0])
        return (x, [np.max(self.dataset.system_manager.total_time_dp())], time.time() - timer)


class HEFT:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = len(self.dataset.svc_set.partitions)

        self.rank = 'rank_u'
        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

    def run_algo(self):
        timer = time.time()
        # calculate rank
        if self.rank == 'rank_u':
            self.system_manager.rank_u = np.zeros(self.num_partitions)
            for partition in self.system_manager.service_set.partitions:
                if len(partition.predecessors) == 0:
                    self.system_manager.calc_rank_u_total_average(partition)
            x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)
            y = np.array([np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(self.num_timeslots)])

        elif self.rank == 'rank_d':
            self.system_manager.rank_d = np.zeros(self.num_partitions)
            for partition in self.system_manager.service_set.partitions:
                if len(partition.successors) == 0:
                    self.system_manager.calc_rank_d_total_average(partition)
            x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)
            y = np.array([np.array(sorted(zip(self.system_manager.rank_d, np.arange(self.num_partitions)), reverse=False), dtype=np.int32)[:,1] for _ in range(self.num_timeslots)])

        # scheduling
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            finish_time, ready_time = None, None
            for _, top_rank in enumerate(y[t]):
                # initialize the earliest finish time of the task
                earliest_finish_time = np.inf
                # for all available server, find earliest finish time server
                for s_id in self.server_lst:
                    if s_id == 0:
                        s_id = self.dataset.partition_device_map[top_rank]
                    temp_x = x[t,top_rank]
                    x[t,top_rank] = s_id
                    self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                    if False not in self.system_manager.constraint_chk():
                        temp_finish_time, temp_ready_time = self.system_manager.get_completion_time_partition(top_rank, deepcopy(finish_time), deepcopy(ready_time))
                        if temp_finish_time[top_rank] < earliest_finish_time:
                            earliest_finish_time = temp_finish_time[top_rank]
                        else:
                            x[t,top_rank] = temp_x
                    else:
                        x[t,top_rank] = temp_x
                self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                finish_time, ready_time = self.system_manager.get_completion_time_partition(top_rank, finish_time, ready_time)
            self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
            # self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)

        x = np.array(x, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        self.system_manager.set_env(deployed_server=x[0], execution_order=y[0])
        return ((x, y), [np.max(self.system_manager.total_time_dp())], time.time() - timer)


class CPOP:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = len(self.dataset.svc_set.partitions)
        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

    def run_algo(self):
        timer = time.time()
        # calculate rank_u and rank_d
        self.system_manager.rank_u = np.zeros(self.num_partitions)
        for partition in self.system_manager.service_set.partitions:
            if len(partition.predecessors) == 0:
                self.system_manager.calc_rank_u_total_average(partition)

        self.system_manager.rank_d = np.zeros(self.num_partitions)
        for partition in self.system_manager.service_set.partitions:
            if len(partition.successors) == 0:
                self.system_manager.calc_rank_d_total_average(partition)

        # find critical path
        rank_sum = self.system_manager.rank_u + self.system_manager.rank_d
        initial_partitions = np.where(self.system_manager.rank_u == np.max(self.system_manager.rank_u))[0]
        # initial_partitions = []
        # for svc_id in range(self.dataset.num_services):
        #     rank_u = np.copy(self.system_manager.rank_u)
        #     rank_u[np.where(self.system_manager.partition_service_map!=svc_id)] = 0
        #     initial_partitions.append(np.argmax(rank_u))
        critical_path = []
        for p_id in initial_partitions:
            critical_path.append(p_id)
            successors = self.dataset.svc_set.partition_successor[p_id]
            while len(successors) > 0:
                idx = np.argmax(rank_sum[successors])
                p_id = successors[idx]
                critical_path.append(p_id)
                successors = self.dataset.svc_set.partition_successor[p_id]

        x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        y = np.array([np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(self.num_timeslots)])

        # scheduling
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            finish_time, ready_time = None, None
            x[t,critical_path] = self.server_lst[-1]
            for _, top_rank in enumerate(y[t]):
                if top_rank in critical_path:
                    x[t,top_rank] = self.server_lst[-1]
                else:
                    # initialize the earliest finish time of the task
                    earliest_finish_time = np.inf
                    # for all available server, find earliest finish time server
                    for s_id in self.server_lst:
                        if s_id == 0:
                            s_id = self.dataset.partition_device_map[top_rank]
                        temp_x = x[t,top_rank]
                        x[t,top_rank] = s_id
                        self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                        if False not in self.system_manager.constraint_chk():
                            temp_finish_time, temp_ready_time = self.system_manager.get_completion_time_partition(top_rank, deepcopy(finish_time), deepcopy(ready_time))
                            if temp_finish_time[top_rank] < earliest_finish_time:
                                earliest_finish_time = temp_finish_time[top_rank]
                            else:
                                x[t,top_rank] = temp_x
                        else:
                            x[t,top_rank] = temp_x
                self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                finish_time, ready_time = self.system_manager.get_completion_time_partition(top_rank, finish_time, ready_time)
            # self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)

        x = np.array(x, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        self.system_manager.set_env(deployed_server=x[0], execution_order=y[0])
        return ((x, y), [np.max(self.system_manager.total_time_dp())], time.time() - timer)


class PEFT:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = len(self.dataset.svc_set.partitions)
        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

    def run_algo(self):
        timer = time.time()
        # calculate optimistic cost table
        num_servers = len(self.server_lst)
        self.system_manager.optimistic_cost_table = np.zeros(shape=(self.num_partitions, num_servers))
        for partition in self.system_manager.service_set.partitions:
            if len(partition.predecessors) == 0:
                for s_idx in range(num_servers):
                    self.system_manager.calc_optimistic_cost_table(partition, s_idx, self.server_lst)
        rank_oct = np.mean(self.system_manager.optimistic_cost_table, axis=1)
        optimistic_cost_table = np.copy(self.system_manager.optimistic_cost_table)

        x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        y = np.array([np.array(sorted(zip(rank_oct, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(self.num_timeslots)])

        # print(y)
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            ready_lst = [partition.total_id for partition in self.system_manager.service_set.partitions if len(partition.predecessors) == 0]
            finish_time, ready_time = None, None
            while len(ready_lst) > 0:
                top_rank = ready_lst.pop(np.argmax(rank_oct[ready_lst]))
                # initialize the earliest finish time of the task
                min_o_eft = np.inf
                # for all available server, find earliest finish time server
                for s_idx, s_id in enumerate(self.server_lst):
                    if s_id == 0:
                        s_id = self.dataset.partition_device_map[top_rank]
                    temp_x = x[t,top_rank]
                    x[t,top_rank] = s_id
                    self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                    if False not in self.system_manager.constraint_chk():
                        temp_finish_time, temp_ready_time = self.system_manager.get_completion_time_partition(top_rank, deepcopy(finish_time), deepcopy(ready_time))
                        o_eft = temp_finish_time[top_rank] + optimistic_cost_table[top_rank, s_idx]
                        if min_o_eft > o_eft:
                            min_o_eft = o_eft
                        else:
                            x[t,top_rank] = temp_x
                    else:
                        x[t,top_rank] = temp_x
                self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                finish_time, ready_time = self.system_manager.get_completion_time_partition(top_rank, finish_time, ready_time)
                
                for succ_id in self.system_manager.service_set.partition_successor[top_rank]:
                    if not 0 in finish_time[self.system_manager.service_set.partition_predecessor[succ_id]]:
                        ready_lst.append(succ_id)
            # self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)

        x = np.array(x, dtype=np.int32)
        y = np.array(y, dtype=np.int32)
        self.system_manager.set_env(deployed_server=x[0], execution_order=y[0])
        return ((x, y), [np.max(self.system_manager.total_time_dp())], time.time() - timer)


class Greedy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = dataset.num_partitions

        self.coarsened_graph = deepcopy(dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
        self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])

        self.rank = 'rank_u'
        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())
    
    def get_uncoarsened_x(self, x):
        result = []
        start = end = 0
        for svc in self.dataset.svc_set.services:
            uncoarsened_x = np.zeros_like(self.coarsened_graph[svc.id])
            start = end
            end += len(self.graph[svc.id])
            for i, x_i in enumerate(x[start:end]):
                uncoarsened_x[np.where(self.coarsened_graph[svc.id]==self.graph[svc.id][i])] = x_i
            result.append(uncoarsened_x)
        return np.concatenate(result, axis=None)

    def init(self):
        self.coarsened_graph = deepcopy(self.dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
        self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])

    def run_algo(self):
        timer = time.time()
        self.init()
        x = np.full(shape=(self.num_pieces,), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        if self.rank == 'rank_oct':
            y = self.system_manager.rank_oct_schedule
            rank = self.system_manager.rank_oct
        else:
            y = self.system_manager.rank_u_schedule
            rank = self.system_manager.rank_u

        partition_piece_map = np.zeros(self.num_partitions, dtype=np.int32)
        idx_start = idx_end = start = end = 0
        for cg in self.coarsened_graph:
            start = end
            end += len(np.unique(cg))
            idx_start = idx_end
            idx_end += len(cg)
            for idx, p in enumerate(cg):
                partition_piece_map[idx_start + idx] = start + p

        piece_rank = [max(rank[np.where(partition_piece_map==idx)]) for idx, x_i in enumerate(x)]
        piece_order = np.array(sorted(zip(piece_rank, np.arange(self.num_pieces)), reverse=True), dtype=np.int32)[:,1]

        # # for piece priority
        # piece_rank = [max(rank[np.where(partition_piece_map==idx)]) for idx, x_i in enumerate(x)]
        # piece_service_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])
        # piece_order = []
        # while sum(piece_rank) > 0:
        #     max_idx = np.argmax(piece_rank)
        #     temp_lst = np.where(piece_service_map==piece_service_map[max_idx])[0]
        #     temp_rank = rank[temp_lst]
        #     for i in reversed(np.argsort(temp_rank)):
        #         piece_order.append(temp_lst[i])
        #         piece_rank[temp_lst[i]] = 0

        for p_id in piece_order:
            minimum_latency = np.inf
            for s_id in self.server_lst:
                if s_id == 0:
                    s_id = self.piece_device_map[p_id]
                temp_x = x[p_id]
                x[p_id] = s_id
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                if self.system_manager.constraint_chk(s_id=s_id):
                    latency = np.max(self.system_manager.total_time_dp())
                    if latency < minimum_latency:
                        minimum_latency = latency
                    else:
                        x[p_id] = temp_x
                else:
                    x[p_id] = temp_x
        print("took: {:.5f}, latency: {:.5f}".format(time.time() - timer, minimum_latency))
        
        l = np.inf
        while l > minimum_latency:
            l = minimum_latency
            for p_id in piece_order:
                minimum_latency = np.inf
                for s_id in self.server_lst:
                    if s_id == 0:
                        s_id = self.piece_device_map[p_id]
                    temp_x = x[p_id]
                    x[p_id] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                    if self.system_manager.constraint_chk(s_id=s_id):
                        latency = np.max(self.system_manager.total_time_dp())
                        if latency < minimum_latency:
                            minimum_latency = latency
                        else:
                            x[p_id] = temp_x
                    else:
                        x[p_id] = temp_x
            print("took: {:.5f}, latency: {:.5f}".format(time.time() - timer, minimum_latency))

        x = np.array([self.get_uncoarsened_x(x) for _ in range(self.num_timeslots)], dtype=np.int32)
        y = np.array([y for _ in range(self.num_timeslots)], dtype=np.int32)
        self.system_manager.set_env(deployed_server=x[0], execution_order=y[0])
        return ((x, y), [np.max(self.system_manager.total_time_dp())], time.time() - timer)