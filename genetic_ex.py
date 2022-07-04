from matplotlib.pyplot import xcorr
import numpy as np
import math
import random
import multiprocessing as mp
import time
from copy import deepcopy

from torch import minimum



class PSOGA:
    def __init__(self, dataset, num_particles, w_max=1.0, w_min=0.5, c1_s=0.5, c1_e=0.5, c2_s=0.5, c2_e=0.5):
        self.num_particles = num_particles
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots

        self.graph = [np.unique(cg) for cg in dataset.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.w_max = w_max            # constant inertia max weight (how much to weigh the previous velocity)
        self.w_min = w_min            # constant inertia min weight (how much to weigh the previous velocity)
        self.c1_s = c1_s                  # cognative constant (start)
        self.c1_e = c1_e                  # cognative constant (end)
        self.c2_s = c2_s                  # social constant (start)
        self.c2_e = c2_e                  # social constant (end)
    
    def get_uncoarsened_x(self, x):
        result = []
        start = end = 0
        for svc in self.dataset.svc_set.services:
            uncoarsened_x = np.zeros_like(self.dataset.coarsened_graph[svc.id])
            start = end
            end += len(self.graph[svc.id])
            for i, x_i in enumerate(x[start:end]):
                uncoarsened_x[np.where(self.dataset.coarsened_graph[svc.id]==self.graph[svc.id][i])] = x_i
            result.append(uncoarsened_x)
        return np.concatenate(result, axis=None)

    def update_w(self, x_t, g_t):
        ps = 0
        for i in range(self.num_particles):
            ps += np.sum(np.equal(x_t[i], g_t), axis=None)
        ps /= self.num_particles * self.num_timeslots * self.num_pieces
        w = self.w_max - (self.w_max - self.w_min) * np.exp(ps / (ps - 1.01))
        return w

    def generate_random_solutions(self, step=2):
        # return np.full(shape=(self.num_particles, self.num_timeslots, self.num_pieces), fill_value=self.server_lst[-1])
        # return np.array([[self.dataset.piece_device_map for _ in range(self.num_timeslots)] for _ in range(0, self.num_particles)])
        # random_solutions = np.zeros((self.num_particles, self.num_timeslots, self.num_pieces))
        # start = end = 0
        # for i in self.server_lst:
        #     start = end
        #     end += step
        #     random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_pieces), fill_value=i)
        # random_solutions[end:self.num_particles,:,:] = np.random.choice(self.server_lst, size=(self.num_particles-end, self.num_timeslots, self.num_pieces))
        random_solutions = np.array([[np.random.choice(self.num_pieces, self.num_pieces, replace=False) for _ in range(self.num_timeslots)] for _ in range(self.num_particles)])
        return random_solutions

    def run_algo(self, loop, verbose=True, local_search=True, random_solution=None):
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        if random_solution is None:
            x_t = self.generate_random_solutions()
        else:
            x_t = random_solution
        p_t, g_t, p_t_eval_lst = self.selection(x_t)

        for i in range(loop):
            v_t = self.mutation(x_t, g_t)
            c_t = self.crossover(v_t, p_t, crossover_rate=(self.c1_e-self.c1_s) * (i / loop) + self.c1_s)
            x_t = self.crossover(c_t, g_t, crossover_rate=(self.c2_e-self.c2_s) * (i / loop) + self.c2_s, is_global=True)
            p_t, g_t, p_t_eval_lst = self.selection(x_t, p_t, g_t, p_t_eval_lst=p_t_eval_lst)
            eval_lst.append(np.max(np.sum(p_t_eval_lst, axis=1)))

            if max_reward < np.max(np.sum(p_t_eval_lst, axis=1)):
                max_reward = np.max(np.sum(p_t_eval_lst, axis=1))
                not_changed_loop = 0
            elif not_changed_loop > 5:
                uncoarsened_x = np.array([self.get_x_from_exec(g_t[t], True) for t in range(self.num_timeslots)])
                self.system_manager.set_env(deployed_server=uncoarsened_x[0])
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(g_t[0])
                print("t:", max(t), t)
                print("e:", sum(e), e)
                print("r:", r, "\033[0m")
                return (uncoarsened_x, eval_lst, time.time() - start)
            else:
                not_changed_loop += 1

            if i == loop - 1:
                uncoarsened_x = np.array([self.get_x_from_exec(g_t[t], True) for t in range(self.num_timeslots)])
                self.system_manager.set_env(deployed_server=uncoarsened_x[0])
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(g_t[0])
                print("t:", max(t), t)
                print("e:", sum(e), e)
                print("r:", r, "\033[0m")

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                print("---------- PSO-GA #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    self.system_manager.set_env(deployed_server=self.get_x_from_exec(g_t[t]))
                    print("#timeslot {} x: {}".format(t, g_t[t]))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(self.system_manager.total_time_dp())
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_x_from_exec(g_t[t]), timeslot=t)
                print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))

        uncoarsened_x = np.array([self.get_x_from_exec(g_t[t], True) for t in range(self.num_timeslots)])
        return (uncoarsened_x, eval_lst, time.time() - start)

    def selection(self, x_t, p_t=None, g_t=None, p_t_eval_lst=None):
        if p_t is None and g_t is None:
            p_t_eval_lst = self.evaluation(x_t)
            p_t_eval_sum = np.sum(p_t_eval_lst, axis=1)
            p_t = np.copy(x_t)
            g_t = np.copy(x_t[np.argmax(p_t_eval_sum),:,:])
        else:
            new_eval_lst = self.evaluation(x_t)
            new_eval_sum = np.sum(new_eval_lst, axis=1)
            p_t_eval_sum = np.sum(p_t_eval_lst, axis=1)
            indices = np.where(new_eval_sum > p_t_eval_sum)
            p_t[indices,:,:] = x_t[indices,:,:]
            p_t_eval_lst[indices,:] = new_eval_lst[indices,:]
            p_t_eval_sum[indices] = new_eval_sum[indices]
            g_t = np.copy(p_t[np.argmax(p_t_eval_sum),:,:])
        return p_t, g_t, p_t_eval_lst

    def crossover_multiprocessing(self, inputs):
        # np.random.seed(random.randint(0,2147483647))
        # a_t, b_t = input
        # for t in range(self.num_timeslots):
        #     a_x = a_t[t]
        #     b_x = b_t[t]

        #     cross_point = np.random.randint(low=1, high=a_x.size - 1)

        #     # crossover x: deployed_server
        #     temp = a_x[:cross_point]
        #     a_x[:cross_point] = b_x[:cross_point]
        #     b_x[:cross_point] = temp
        #     a_t[t,:self.num_pieces] = a_x
        #     b_t[t,:self.num_pieces] = b_x
        # if random.random() > 0.5:
        #     return a_t
        # else:
        #     return b_t
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t = inputs
        for t in range(self.num_timeslots):
            a_x = a_t[t]
            b_x = b_t[t]
            cross_points = np.random.randint(low=0, high=a_x.size, size=2)

            # # crossover x: deployed_server
            # a_x[cross_point[0]:cross_point[1]] = b_x[cross_point[0]:cross_point[1]]
            # a_t[t,:self.num_pieces] = a_x

            # crossover x: execution_order
            for cross_point in cross_points:
                a_x[np.where(a_x==b_x[cross_point])] = a_x[cross_point]
                a_x[cross_point] = b_x[cross_point]
            a_t[t,:self.num_pieces] = a_x
        return a_t

    def crossover(self, a_t, b_t, crossover_rate, is_global=False):
        new_a_t = np.copy(a_t)
        crossover_idx = np.random.rand(self.num_particles)
        crossover_idx = crossover_idx < crossover_rate
        crossover_idx = np.where(crossover_idx)[0]
        if len(crossover_idx) == 0:
            crossover_idx = np.array([np.random.randint(low=0, high=self.num_particles)])
        
        if is_global:
            temp = [self.crossover_multiprocessing((new_a_t[i], b_t)) for i in crossover_idx]
        else:
            temp = [self.crossover_multiprocessing((new_a_t[i], b_t[i])) for i in crossover_idx]
        new_a_t[crossover_idx] = np.array(temp)
        return new_a_t
        if is_global:
            working_queue = [(new_a_t[i], b_t) for i in crossover_idx]
        else:
            working_queue = [(new_a_t[i], b_t[i]) for i in crossover_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.crossover_multiprocessing, working_queue))
        new_a_t[crossover_idx] = np.array(temp)
        return new_a_t

    def mutation_multiprocessing(self, v_t):
        np.random.seed(random.randint(0,2147483647))
        for t in range(self.num_timeslots):
            x = v_t[t]
            mutation_point = np.random.randint(low=0, high=self.num_pieces, size=2)

            # mutate x: execution_order
            temp = x[mutation_point[0]]
            x[mutation_point[0]] = x[mutation_point[1]]
            x[mutation_point[1]] = temp
            v_t[t,:self.num_pieces] = x
        return v_t
        np.random.seed(random.randint(0,2147483647))
        for t in range(self.num_timeslots):
            x = v_t[t]
            mutation_point = np.random.randint(low=0, high=self.num_pieces)

            # mutate x: deployed_server
            another_s_id = np.random.choice(self.server_lst)
            x[mutation_point] = another_s_id
            v_t[t,:self.num_pieces] = x
        return v_t

    def mutation(self, x_t, g_t, mutation_ratio=None):
        v_t = np.copy(x_t)
        if mutation_ratio == None:
            w = self.update_w(v_t[:,:,:self.num_pieces], g_t[:,:self.num_pieces])
        else:
            w = mutation_ratio
        mutation_idx = np.random.rand(self.num_particles)
        mutation_idx = mutation_idx < w
        mutation_idx = np.where(mutation_idx)[0]
        if len(mutation_idx) == 0:
            mutation_idx = np.array([np.random.randint(low=0, high=self.num_particles)])
        
        temp = [self.mutation_multiprocessing(v_t[i]) for i in mutation_idx]
        v_t[mutation_idx] = np.array(temp)
        return v_t
        working_queue = [v_t[i] for i in mutation_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.mutation_multiprocessing, working_queue))
        v_t[mutation_idx] = np.array(temp)
        return v_t
    
    def get_x_from_exec(self, piece_order, local_search=False):
        x = np.full(shape=(self.num_pieces,), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        for p_id in piece_order:
            # initialize the earliest finish time of the task
            minimum_latency = np.inf
            # for all available server, find earliest finish time server
            for s_id in self.server_lst:
                if s_id == 0:
                    s_id = self.dataset.piece_device_map[p_id]
                temp_x = x[p_id]
                x[p_id] = s_id
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                if self.system_manager.constraint_chk(s_id=s_id):
                    latency = max(self.system_manager.total_time_dp())
                    if latency < minimum_latency:
                        minimum_latency = latency
                    else:
                        x[p_id] = temp_x
                else:
                    x[p_id] = temp_x
        uncoarsened_x = self.get_uncoarsened_x(x)

        if local_search:
            print(x)
            uncoarsened_x = self.local_search(uncoarsened_x)
        return uncoarsened_x

    def local_search(self, x):
        start = time.time()
        self.system_manager.set_env(deployed_server=x)
        latency = self.system_manager.total_time_dp()
        minimum_latency = max(latency)
        svc_lst = np.argsort(latency)
        print("local search before:", minimum_latency)
        for svc in svc_lst:
            # ranks = self.system_manager.rank_oct_schedule[np.where(self.system_manager.partition_service_map == svc)]
            ranks = self.system_manager.rank_u_schedule[np.where(self.system_manager.partition_service_map == svc)]
            for top_rank in ranks:
                # sibling's server
                neighbor_partitions = np.where(self.system_manager.partition_layer_map == self.system_manager.partition_layer_map[top_rank])[0]
                neighbor_partitions = neighbor_partitions[np.where((neighbor_partitions == top_rank + 1) | (neighbor_partitions == top_rank - 1))]
                server_lst = set(x[neighbor_partitions]) - set([x[top_rank]])
                # for all available server, find earliest finish time server
                for s_id in server_lst:
                    temp_x = x[top_rank]
                    x[top_rank] = s_id
                    self.system_manager.set_env(deployed_server=x)
                    if False not in self.system_manager.constraint_chk():
                        latency = max(self.system_manager.total_time_dp())
                        if latency < minimum_latency:
                            minimum_latency = latency
                        else:
                            x[top_rank] = temp_x
                    else:
                        x[top_rank] = temp_x
        print("took:", time.time() - start)
        return x
    
    def evaluation_multiprocessing(self, postition):
        self.system_manager.init_env()
        reward = []
        for t in range(self.num_timeslots):
            self.system_manager.set_env(deployed_server=self.get_x_from_exec(postition[t]))
            reward.append(self.system_manager.get_reward())
            self.system_manager.after_timeslot(deployed_server=self.get_x_from_exec(postition[t]), timeslot=t)
        return reward

    def evaluation(self, postitions):
        working_queue = [postition for postition in postitions]
        with mp.Pool(processes=30) as pool:
            evaluation_lst = list(pool.map(self.evaluation_multiprocessing, working_queue))
        evaluation_lst = np.array(evaluation_lst)
        return evaluation_lst


class Genetic(PSOGA):
    def __init__(self, dataset, num_solutions, mutation_ratio=0.3, cross_over_ratio=0.7):
        self.num_solutions = num_solutions
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots

        self.graph = [np.unique(cg) for cg in dataset.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.mutation_ratio = mutation_ratio
        self.cross_over_ratio = cross_over_ratio

    def generate_random_solutions(self, step=2):
        # return np.full(shape=(self.num_solutions, self.num_timeslots, self.num_pieces), fill_value=self.server_lst[-1])
        # return np.array([[self.dataset.piece_device_map for _ in range(self.num_timeslots)] for _ in range(0, self.num_solutions)])
        # random_solutions = np.zeros((self.num_solutions, self.num_timeslots, self.num_pieces))
        # start = end = 0
        # for i in self.server_lst:
        #     start = end
        #     end += step
        #     random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_pieces), fill_value=i)
        # random_solutions[end:self.num_solutions,:,:] = np.random.choice(self.server_lst, size=(self.num_solutions-end, self.num_timeslots, self.num_pieces))
        random_solutions = np.array([[np.random.choice(self.num_pieces, self.num_pieces, replace=False) for _ in range(self.num_timeslots)] for _ in range(self.num_solutions)])
        return random_solutions

    def run_algo(self, loop, verbose=True, local_search=True, random_solution=None):
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        if random_solution is None:
            p_t = self.generate_random_solutions()
        else:
            p_t = random_solution

        p_known = np.copy(p_t)

        for i in range(loop):
            q_t = self.selection(p_t, p_known)
            q_t = self.mutation(q_t, self.mutation_ratio)
            q_t = self.crossover(q_t, self.cross_over_ratio)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(p_t, q_t)
            eval_lst.append(v)

            if max_reward < v:
                max_reward = v
                not_changed_loop = 0
            elif not_changed_loop > 5:
                uncoarsened_x = np.array([self.get_x_from_exec(p_t[0,t], True) for t in range(self.num_timeslots)])
                self.system_manager.set_env(deployed_server=uncoarsened_x[0])
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(p_t[0,0])
                print("t:", max(t), t)
                print("e:", sum(e), e)
                print("r:", r, "\033[0m")
                return (uncoarsened_x, eval_lst, time.time() - start)
            else:
                not_changed_loop += 1

            if i == loop - 1:
                uncoarsened_x = np.array([self.get_x_from_exec(p_t[0,t], True) for t in range(self.num_timeslots)])
                self.system_manager.set_env(deployed_server=uncoarsened_x[0])
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(p_t[0,0])
                print("t:", max(t), t)
                print("e:", sum(e), e)
                print("r:", r, "\033[0m")

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                if local_search:
                    print("---------- Memetic #{} loop ----------".format(i))
                else:
                    print("---------- Genetic #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    self.system_manager.set_env(deployed_server=self.get_x_from_exec(p_t[0,t]))
                    print("#timeslot {} x: {}".format(t, p_t[0,t]))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(self.system_manager.total_time_dp())
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(p_t[0,t]), timeslot=t)
                print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))

        uncoarsened_x = np.array([self.get_x_from_exec(p_t[0,t], True) for t in range(self.num_timeslots)])
        return (uncoarsened_x, eval_lst, time.time() - start)

    @staticmethod
    def union(*args):
        if len(args) == 0:
            union = np.array([])
            max_len = 0
        elif len(args) == 1:
            union = args[0]
            max_len = len(args[0])
        else:
            union = np.append(args[0], args[1], axis=0)
            max_len = max(len(args[0]), len(args[1]))

        for i in range(2, len(args)):
            if max_len < len(args[i]):
                max_len = len(args[i])
            union = np.append(union, args[i], axis=0)
        return union, max_len

    def selection(self, *args):
        union, max_len = self.union(*args)
        order = np.arange(union.shape[0])
        np.random.shuffle(order)
        return union[order[:max_len], :]

    def fitness_selection(self, *args):
        union, max_len = self.union(*args)
        ev_lst = np.sum(self.evaluation(union), axis=1)
        ev_lst = list(map(lambda x: (x[0], x[1]), enumerate(ev_lst)))
        ev_lst.sort(key=lambda x: x[1], reverse=True)
        sorted_idx = list(map(lambda x: x[0], ev_lst))
        union = union[sorted_idx[:max_len], :]
        return union, ev_lst[0][1]

    def crossover_multiprocessing(self, inputs):
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t = inputs
        for t in range(self.num_timeslots):
            a_x = a_t[t]
            b_x = b_t[t]

            cross_points = np.random.randint(low=1, high=a_x.size - 1, size=2)

            # crossover x: execution_order
            for cross_point in cross_points:
                temp_a = a_x[cross_point]
                temp_b = b_x[cross_point]
                a_x[np.where(a_x==temp_b)] = temp_a
                b_x[np.where(b_x==temp_a)] = temp_b
                a_x[cross_point] = temp_b
                b_x[cross_point] = temp_a
            a_t[t,:self.num_pieces] = a_x
            b_t[t,:self.num_pieces] = b_x
        action = np.concatenate([a_t.reshape(1,self.num_timeslots,-1), b_t.reshape(1,self.num_timeslots,-1)], axis=0)
        return action
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t = input
        for t in range(self.num_timeslots):
            a_x = a_t[t]
            b_x = b_t[t]

            cross_point = np.random.randint(low=1, high=a_x.size - 1)

            # crossover x: deployed_server
            temp = a_x[:cross_point]
            a_x[:cross_point] = b_x[:cross_point]
            b_x[:cross_point] = temp
            a_t[t,:self.num_pieces] = a_x
            b_t[t,:self.num_pieces] = b_x
        action = np.concatenate([a_t.reshape(1,self.num_timeslots,-1), b_t.reshape(1,self.num_timeslots,-1)], axis=0)
        return action

    def crossover(self, action, crossover_rate):
        crossover_idx = np.random.rand(self.num_solutions)
        crossover_idx = crossover_idx < crossover_rate
        crossover_idx = np.where(crossover_idx)[0]
        if len(crossover_idx) < 2:
            crossover_idx = np.array([np.random.randint(low=0, high=self.num_solutions, size=2)])
        np.random.shuffle(crossover_idx)
        if len(crossover_idx) - math.floor(crossover_idx.size / 2) * 2:
            crossover_idx = crossover_idx[:-1]
        
        temp = [self.crossover_multiprocessing((action[crossover_idx[i * 2]], action[crossover_idx[i * 2 + 1]])) for i in range(math.floor(crossover_idx.size / 2))]
        action[crossover_idx] = np.concatenate(temp, axis=0)
        return action
        working_queue = [(action[crossover_idx[i * 2]], action[crossover_idx[i * 2 + 1]]) for i in range(math.floor(crossover_idx.size / 2))]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.crossover_multiprocessing, working_queue))
        action[crossover_idx] = np.concatenate(temp, axis=0)
        return action

    def mutation_multiprocessing(self, action):
        np.random.seed(random.randint(0,2147483647))
        for t in range(self.num_timeslots):
            mutation_point = np.random.randint(low=0, high=self.num_pieces, size=2)

            # mutate x: execution_order
            temp = action[t,mutation_point[0]]
            action[t,mutation_point[0]] = action[t,mutation_point[1]]
            action[t,mutation_point[1]] = temp
        return action.reshape(1,self.num_timeslots,-1)
        np.random.seed(random.randint(0,2147483647))
        for t in range(self.num_timeslots):
            mutation_point = np.random.randint(low=0, high=self.num_pieces)

            # mutate x: deployed_server
            another_s_id = np.random.choice(self.server_lst)
            action[t,mutation_point] = another_s_id
        return action.reshape(1,self.num_timeslots,-1)

    def mutation(self, action, mutation_ratio):
        mutation_idx = np.random.rand(self.num_solutions)
        mutation_idx = mutation_idx < mutation_ratio
        mutation_idx = np.where(mutation_idx)[0]
        if len(mutation_idx) == 0:
            mutation_idx = np.array([np.random.randint(low=0, high=self.num_solutions)])
        
        temp = [self.mutation_multiprocessing(action[i]) for i in mutation_idx]
        action[mutation_idx] = np.concatenate(temp, axis=0)
        return action
        working_queue = [action[i] for i in mutation_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.mutation_multiprocessing, working_queue))
        action[mutation_idx] = np.concatenate(temp, axis=0)
        return action


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

        # print(y)
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            finish_time, ready_time = None, None
            for _, top_rank in enumerate(y[t]):
                # initialize the earliest finish time of the task
                earliest_finish_time = np.inf
                # for all available server, find earliest finish time server
                for s_id in self.server_lst:
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
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


class CPOP:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = len(self.dataset.svc_set.partitions)
        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

    def run_algo(self):
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

        # print(y)
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            finish_time, ready_time = None, None
            for _, top_rank in enumerate(y[t]):
                if top_rank in critical_path:
                    x[t,top_rank] = self.server_lst[-1]
                else:
                    # initialize the earliest finish time of the task
                    earliest_finish_time = np.inf
                    # for all available server, find earliest finish time server
                    for s_id in self.server_lst:
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
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


class PEFT:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = len(self.dataset.svc_set.partitions)
        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

    def run_algo(self):
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
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


class Greedy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = len(self.dataset.svc_set.partitions)

        self.graph = [np.unique(cg) for cg in dataset.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])

        self.rank = 'rank_u'
        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())
    
    def get_uncoarsened_x(self, x):
        result = []
        start = end = 0
        for svc in self.dataset.svc_set.services:
            uncoarsened_x = np.zeros_like(self.dataset.coarsened_graph[svc.id])
            start = end
            end += len(self.graph[svc.id])
            for i, x_i in enumerate(x[start:end]):
                uncoarsened_x[np.where(self.dataset.coarsened_graph[svc.id]==self.graph[svc.id][i])] = x_i
            result.append(uncoarsened_x)
        return np.concatenate(result, axis=None)

    def run_algo(self):
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

        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            finish_time, ready_time = None, None
            for _, top_rank in enumerate(y[t]):
                # initialize the earliest finish time of the task
                earliest_finish_time = np.inf
                # for all available server, find earliest finish time server
                for s_id in self.server_lst:
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
            minimum_latency = last_latency = max(self.system_manager.total_time_dp())
            loop = 0
            while True:
                # while loop memory leak
                working_queue = [(x[t], y[t], minimum_latency, True)]
                with mp.Pool(processes=1) as pool:
                    temp = list(pool.map(self.local_search, working_queue))
                x[t], minimum_latency = temp[0]

                working_queue = [(x[t], y[t], minimum_latency, False)]
                with mp.Pool(processes=1) as pool:
                    temp = list(pool.map(self.local_search, working_queue))
                x[t], minimum_latency = temp[0]

                loop += 1
                if last_latency > minimum_latency:
                    last_latency = minimum_latency
                else:
                    break
                print("loop", loop, minimum_latency)
            print("loop", loop, minimum_latency)
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def local_search(self, inputs):
        x, y, minimum_latency, reverse = inputs
        service = np.argmax(self.system_manager.total_time_dp())
        if reverse:
            for top_rank in reversed(y[np.where(self.dataset.partition_service_map==service)]):
                if random.random() > 1.:
                    continue
                # for all available server, find earliest finish time server
                for s_id in self.server_lst:
                    temp_x = x[top_rank]
                    x[top_rank] = s_id
                    self.system_manager.set_env(deployed_server=x, execution_order=y)
                    if False not in self.system_manager.constraint_chk():
                        latency = max(self.system_manager.total_time_dp())
                        if latency < minimum_latency:
                            minimum_latency = latency
                        else:
                            x[top_rank] = temp_x
                    else:
                        x[top_rank] = temp_x
        else:
            for top_rank in y[np.where(self.dataset.partition_service_map==service)]:
                if random.random() > 1.:
                    continue
                # for all available server, find earliest finish time server
                for s_id in self.server_lst:
                    temp_x = x[top_rank]
                    x[top_rank] = s_id
                    self.system_manager.set_env(deployed_server=x, execution_order=y)
                    if False not in self.system_manager.constraint_chk():
                        latency = max(self.system_manager.total_time_dp())
                        if latency < minimum_latency:
                            minimum_latency = latency
                        else:
                            x[top_rank] = temp_x
                    else:
                        x[top_rank] = temp_x
        return (x, minimum_latency)

    def run_algo_piecewise(self):
        uncoarsened_x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)

        if self.rank == 'rank_u':
            y = np.array([self.system_manager.rank_u_schedule for _ in range(self.num_timeslots)])
        elif self.rank == 'rank_d':
            y = np.array([self.system_manager.rank_d_schedule for _ in range(self.num_timeslots)])

        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.system_manager.set_env(deployed_server=uncoarsened_x[t])
            minimum_latency = last_latency = np.inf
            working_queue = [i for i in range(128)]
            for _ in range(5):
                with mp.Pool(processes=30) as pool:
                    temp = list(pool.map(self.local_search_piecewise, working_queue))
                for (x, minimum_latency) in temp:
                    if last_latency > minimum_latency:
                        last_latency = minimum_latency
                        uncoarsened_x[t] = self.get_uncoarsened_x(x)

            self.system_manager.set_env(deployed_server=uncoarsened_x[t])
            minimum_latency = last_latency = max(self.system_manager.total_time_dp())
            loop = 0
            print("loop", loop, minimum_latency)
            while True:
                # while loop memory leak
                working_queue = [(uncoarsened_x[t], y[t], minimum_latency, True)]
                with mp.Pool(processes=1) as pool:
                    temp = list(pool.map(self.local_search, working_queue))
                uncoarsened_x[t], minimum_latency = temp[0]

                working_queue = [(uncoarsened_x[t], y[t], minimum_latency, False)]
                with mp.Pool(processes=1) as pool:
                    temp = list(pool.map(self.local_search, working_queue))
                uncoarsened_x[t], minimum_latency = temp[0]

                loop += 1
                if last_latency > minimum_latency and loop <= 1:
                    last_latency = minimum_latency
                else:
                    break
                print("loop", loop, minimum_latency)
            print("loop", loop, minimum_latency)
            self.system_manager.after_timeslot(deployed_server=uncoarsened_x[t], timeslot=t)
        return np.array([uncoarsened_x[t] for t in range(self.num_timeslots)])

    def local_search_piecewise(self, pid):
        x = np.full(shape=(self.num_pieces,), fill_value=self.system_manager.cloud_id, dtype=np.int32)

        piece_order = list(range(self.num_pieces))
        random.shuffle(piece_order)

        for p_id in piece_order:
            # initialize the earliest finish time of the task
            minimum_latency = np.inf
            # for all available server, find earliest finish time server
            for s_id in self.server_lst:
                if s_id == 0:
                    s_id = self.dataset.piece_device_map[p_id]
                temp_x = x[p_id]
                x[p_id] = s_id
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                if self.system_manager.constraint_chk(s_id=s_id):
                    latency = max(self.system_manager.total_time_dp())
                    if latency < minimum_latency:
                        minimum_latency = latency
                    else:
                        x[p_id] = temp_x
                else:
                    x[p_id] = temp_x
        return (x, minimum_latency)