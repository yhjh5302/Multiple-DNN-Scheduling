import numpy as np
import math
import random
from dag_data_generator import DAGDataSet

import multiprocessing as mp
import time


class PSOGA:
    def __init__(self, dataset, num_particles, w_max=1.0, w_min=0.5, c1=0.5, c2=0.5):
        self.num_particles = num_particles
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots

        self.graph = np.unique(dataset.coarsened_graph)
        self.num_partitions = len(np.unique(dataset.coarsened_graph))
        self.uncoarsened_graph = dataset.coarsened_graph
        self.num_uncoarsened_partitons = dataset.num_partitions

        self.scheduling_lst = np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_uncoarsened_partitons)), reverse=True), dtype=np.int32)[:,1]

        self.w_max = w_max            # constant inertia max weight (how much to weigh the previous velocity)
        self.w_min = w_min            # constant inertia min weight (how much to weigh the previous velocity)
        self.c1 = c1                  # cognative constant
        self.c2 = c2                  # social constant
    
    def get_uncoarsened_x(self, x):
        uncoarsened_x = np.zeros(self.num_uncoarsened_partitons)
        for i, x_i in enumerate(x):
            uncoarsened_x[np.where(self.uncoarsened_graph==self.graph[i])] = x_i
        return uncoarsened_x

    def update_w(self, x_t, g_t):
        ps = 0
        for i in range(self.num_particles):
            ps += np.sum(np.equal(x_t[i], g_t), axis=None)
        ps /= self.num_particles * self.num_timeslots * self.num_partitions
        w = self.w_max - (self.w_max - self.w_min) * np.exp(ps / (ps - 1.01))
        return w
    
    @staticmethod
    def generate_random_order(size):
        temp = list(range(size))
        random.shuffle(temp)
        return temp

    def generate_random_solutions(self):
        return np.random.randint(low=self.num_servers-2, high=self.num_servers-1, size=(self.num_particles, self.num_timeslots, self.num_partitions))
    
    def local_search_multiprocessing(self, action):
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(action[t])

            for j in np.random.choice(self.num_partitions, size=1, replace=False): # for jth layer
                # local search x: deployed_server
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst)
                min_time = sum(self.system_manager.total_time_dp())
                for s_id in range(self.num_servers-1):
                    if s_id == action[t,j]:
                        continue
                    temp = action[t,j]
                    action[t,j] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst)
                    cur_time = sum(self.system_manager.total_time_dp())
                    if cur_time < min_time and all(self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst)):
                        min_time = cur_time
                    else:
                        action[t,j] = temp

            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst, timeslot=t)
        return action

    # we have to find local optimum from current chromosomes.
    def local_search(self, action, local_prob=0.2):
        local_idx = np.random.rand(action.shape[0])
        local_idx = local_idx < local_prob
        local_idx = np.where(local_idx)[0]
        np.random.shuffle(local_idx)

        working_queue = [action[i] for i in local_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.local_search_multiprocessing, working_queue))
        action[local_idx] = np.array(temp)
        return action

    def run_algo(self, loop, verbose=True, local_search=True):
        x_t = self.generate_random_solutions()
        x_t = self.reparation(x_t)
        if local_search:
            x_t = self.local_search(x_t, local_prob=0.2)
        p_t, g_t, p_t_eval_lst = self.selection(x_t)

        for i in range(loop):
            start = time.time()
            v_t = self.mutation(x_t, g_t)
            c_t = self.crossover(v_t, p_t, crossover_rate=self.c1)
            x_t = self.crossover(c_t, g_t, crossover_rate=self.c2, is_global=True)
            x_t = self.reparation(x_t)
            
            if local_search:
                x_t = self.local_search(x_t, local_prob=0.2)
            p_t, g_t, p_t_eval_lst = self.selection(x_t, p_t, g_t, p_t_eval_lst=p_t_eval_lst)

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                print("---------- #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(g_t[t]), execution_order=self.scheduling_lst)
                    print("#timeslot {} x: {}".format(t, g_t[t]))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(sum(self.system_manager.total_time_dp()))
                    total_energy.append(sum([s.energy_consumption() for s in self.system_manager.local.values()]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(g_t[t]), execution_order=self.scheduling_lst, timeslot=t)
                print("mean t: {:.5f}".format(sum(total_time) / self.num_timeslots))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("took: {:.5f} sec".format(end - start))
        return np.array([self.get_uncoarsened_x(g_t[t]) for t in range(self.num_timeslots)]), np.array([self.scheduling_lst for t in range(self.num_timeslots)])

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

    def crossover_multiprocessing(self, input):
        a_t, b_t = input
        for t in range(self.num_timeslots):
            a_x = a_t[t]
            b_x = b_t[t]
            cross_point = np.random.randint(low=0, high=a_x.size, size=2)

            # crossover x: deployed_server
            a_x[cross_point[0]:cross_point[1]] = b_x[cross_point[0]:cross_point[1]]
            a_t[t,:self.num_partitions] = a_x
        return a_t

    def crossover(self, a_t, b_t, crossover_rate, is_global=False):
        new_a_t = np.copy(a_t)
        crossover_idx = np.random.rand(self.num_particles)
        crossover_idx = crossover_idx < crossover_rate
        crossover_idx = np.where(crossover_idx)[0]
        
        if is_global:
            working_queue = [(new_a_t[i], b_t) for i in crossover_idx]
        else:
            working_queue = [(new_a_t[i], b_t[i]) for i in crossover_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.crossover_multiprocessing, working_queue))
        new_a_t[crossover_idx] = np.array(temp)
        return new_a_t

    def mutation_multiprocessing(self, v_t):
        for t in range(self.num_timeslots):
            x = v_t[t]

            mutation_point = np.random.randint(low=0, high=self.num_partitions)

            # mutate x: deployed_server
            x[mutation_point] = np.random.randint(low=0, high=self.num_servers-1)
            v_t[t,:self.num_partitions] = x
        return v_t

    def mutation(self, x_t, g_t, mutation_ratio=None):
        v_t = np.copy(x_t)
        if mutation_ratio == None:
            w = self.update_w(v_t[:,:,:self.num_partitions], g_t[:,:self.num_partitions])
        else:
            w = mutation_ratio
        mutation_idx = np.random.rand(self.num_particles)
        mutation_idx = mutation_idx < w
        mutation_idx = np.where(mutation_idx)[0]
        
        working_queue = [v_t[i] for i in mutation_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.mutation_multiprocessing, working_queue))
        v_t[mutation_idx] = np.array(temp)
        return v_t

    # convert invalid action to valid action, deployed server action.
    def deployed_server_reparation(self, x):
        server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())
        cloud_lst = list(self.system_manager.cloud.keys())
        while False in self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x), execution_order=self.scheduling_lst):
            # 각 서버에 대해서,
            for s_id in range(self.num_servers-1):
                deployed_container_lst = list(np.where(x == s_id)[0])
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                while self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x), execution_order=self.scheduling_lst, s_id=s_id) == False:
                    # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        if s_id != another_s_id and self.system_manager.server[another_s_id].cur_energy > 0 and self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x), execution_order=self.scheduling_lst, s_id=another_s_id):
                            x[c_id] = another_s_id # 한번 넣어보고
                            if self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x), execution_order=self.scheduling_lst, s_id=another_s_id): # 자원 넘치는지 확인.
                                break
                            else:
                                x[c_id] = s_id # 자원 넘치면 롤백

    # convert invalid action to valid action, multiprocessing function.
    def reparation_multiprocessing(self, postition):
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(postition[t])
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(postition[t]), execution_order=self.scheduling_lst, timeslot=t)
        return postition

    # convert invalid action to valid action.
    def reparation(self, postitions):
        working_queue = [position for position in postitions]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.reparation_multiprocessing, working_queue))
        postitions = np.array(temp)
        return postitions
    
    def evaluation_multiprocessing(self, postition):
        self.system_manager.init_env()
        reward = []
        for t in range(self.num_timeslots):
            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(postition[t]), execution_order=self.scheduling_lst)
            reward.append(self.system_manager.get_reward())
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(postition[t]), execution_order=self.scheduling_lst, timeslot=t)
        return reward

    def evaluation(self, postitions):
        working_queue = [postition for postition in postitions]
        with mp.Pool(processes=30) as pool:
            evaluation_lst = list(pool.map(self.evaluation_multiprocessing, working_queue))
        evaluation_lst = np.array(evaluation_lst)
        return evaluation_lst


class Genetic(PSOGA):
    def __init__(self, dataset, num_solutions, mutation_ratio=0.5, cross_over_ratio=0.5):
        self.num_solutions = num_solutions
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots

        self.graph = np.unique(dataset.coarsened_graph)
        self.num_partitions = len(np.unique(dataset.coarsened_graph))
        self.uncoarsened_graph = dataset.coarsened_graph
        self.num_uncoarsened_partitons = dataset.num_partitions

        self.scheduling_lst = np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_uncoarsened_partitons)), reverse=True), dtype=np.int32)[:,1]

        self.mutation_ratio = mutation_ratio
        self.cross_over_ratio = cross_over_ratio

    def generate_random_solutions(self):
        return np.random.randint(low=self.num_servers-2, high=self.num_servers-1, size=(self.num_solutions, self.num_timeslots, self.num_partitions))

    def run_algo(self, loop, verbose=True, local_search=True):
        ev_lst = np.zeros(loop, dtype=np.float_)
        p_t = self.generate_random_solutions()
        p_t = self.reparation(p_t)

        p_known = np.copy(p_t)
        if local_search:
            p_t = self.local_search(p_t, local_prob=0.2)

        for i in range(loop):
            start = time.time()
            q_t = self.selection(p_t, p_known)

            q_t = self.crossover(q_t, self.cross_over_ratio)
            q_t = self.mutation(q_t, self.mutation_ratio)
            q_t = self.reparation(q_t)

            if local_search:
                q_t = self.local_search(q_t, local_prob=0.2)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(p_t, q_t)
            ev_lst[i] = v

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                print("---------- #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(p_t[0,t]), execution_order=self.scheduling_lst)
                    print("#timeslot {} x: {}".format(t, p_t[0,t]))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(sum(self.system_manager.total_time_dp()))
                    total_energy.append(sum([s.energy_consumption() for s in self.system_manager.local.values()]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(p_t[0,t]), execution_order=self.scheduling_lst, timeslot=t)
                print("mean t: {:.5f}".format(sum(total_time) / self.num_timeslots))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("took: {:.5f} sec".format(end - start))
        return np.array([self.get_uncoarsened_x(p_t[0,t]) for t in range(self.num_timeslots)]), np.array([self.scheduling_lst for t in range(self.num_timeslots)])

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
        ev_lst = np.reciprocal(np.sum(np.reciprocal(self.evaluation(union)), axis=1))
        ev_lst = list(map(lambda x: (x[0], x[1]), enumerate(ev_lst)))
        ev_lst.sort(key=lambda x: x[1], reverse=True)
        sorted_idx = list(map(lambda x: x[0], ev_lst))
        union = union[sorted_idx[:max_len], :]
        return union, ev_lst[0][1]

    def crossover_multiprocessing(self, input):
        a_t, b_t = input
        for t in range(self.num_timeslots):
            a_x = a_t[t]
            b_x = b_t[t]

            cross_point = np.random.randint(low=1, high=a_x.size - 1)

            # crossover x: deployed_server
            temp = a_x[:cross_point]
            a_x[:cross_point] = b_x[:cross_point]
            b_x[:cross_point] = temp
            a_t[t,:self.num_partitions] = a_x
            b_t[t,:self.num_partitions] = b_x
        action = np.concatenate([a_t.reshape(1,self.num_timeslots,-1), b_t.reshape(1,self.num_timeslots,-1)], axis=0)
        return action

    def crossover(self, action, crossover_rate):
        crossover_idx = np.random.rand(self.num_solutions)
        crossover_idx = crossover_idx < crossover_rate
        crossover_idx = np.where(crossover_idx)[0]
        np.random.shuffle(crossover_idx)
        if len(crossover_idx) - math.floor(crossover_idx.size / 2) * 2:
            crossover_idx = crossover_idx[:-1]
        
        working_queue = [(action[crossover_idx[i * 2]], action[crossover_idx[i * 2 + 1]]) for i in range(math.floor(crossover_idx.size / 2))]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.crossover_multiprocessing, working_queue))
        action[crossover_idx] = np.concatenate(temp, axis=0)
        return action

    def mutation_multiprocessing(self, action):
        for t in range(self.num_timeslots):
            mutation_point = np.random.randint(low=0, high=self.num_partitions)

            # mutate x: deployed_server
            action[t,mutation_point] = np.random.randint(low=0, high=self.num_servers-1)
        return action.reshape(1,self.num_timeslots,-1)

    def mutation(self, action, mutation_ratio):
        mutation_idx = np.random.rand(self.num_solutions)
        mutation_idx = mutation_idx < mutation_ratio
        mutation_idx = np.where(mutation_idx)[0]
        
        working_queue = [action[i] for i in mutation_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.mutation_multiprocessing, working_queue))
        action[mutation_idx] = np.concatenate(temp, axis=0)
        return action


class HEFT:
    def __init__(self, dataset):
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_partitions = dataset.num_partitions
        self.num_timeslots = dataset.num_timeslots

    def run_algo(self):
        x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        y = np.array([np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(self.num_timeslots)])

        server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            Tr = Tf = None
            for _, top_rank in enumerate(y[t]):
                # initialize the earliest finish time of the task
                earliest_finish_time = np.inf
                # for all available server, find earliest finish time server
                for s_id in server_lst:
                    temp_x = x[t,top_rank]
                    x[t,top_rank] = s_id
                    if self.system_manager.constraint_chk(x[t], y[t], s_id):
                        self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                        finish_time, temp_Tr, temp_Tf = self.system_manager.get_completion_time(top_rank, Tr, Tf)
                        if finish_time < earliest_finish_time:
                            earliest_finish_time = finish_time
                        else:
                            x[t,top_rank] = temp_x
                    else:
                        x[t,top_rank] = temp_x
                Tr = temp_Tr
                Tf = temp_Tf
            
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def result(x_lst, y_lst, took, algorithm_name):
    total_time = []
    total_energy = []
    total_reward = []
    print("---------- " + algorithm_name + " ----------")
    dataset.system_manager.init_env()
    for t in range(dataset.num_timeslots):
        x = np.array(x_lst[t])
        y = np.array(y_lst[t])
        dataset.system_manager.set_env(deployed_server=x, execution_order=y)
        print("#timeslot {} x: {}".format(t, x))
        print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in dataset.system_manager.server.values()]))
        #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in dataset.system_manager.server.values()]))
        #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in dataset.system_manager.server.values()]))
        #print("#timeslot {} t: {}".format(t, dataset.system_manager.total_time_dp()))
        total_time.append(sum(dataset.system_manager.total_time_dp()))
        total_energy.append(sum([s.energy_consumption() for s in dataset.system_manager.local.values()]))
        total_reward.append(dataset.system_manager.get_reward())
        dataset.system_manager.after_timeslot(deployed_server=x, execution_order=y, timeslot=t)
    print("mean t: {:.5f}".format(sum(total_time) / dataset.num_timeslots))
    print("mean e: {:.5f}".format(sum(total_energy) / dataset.num_timeslots))
    print("mean r: {:.5f}".format(sum(total_reward) / dataset.num_timeslots))
    print("took: {:.5f} sec".format(took))
    print("---------- Greedy Algorithm ----------\n")
    return total_time, total_energy, total_reward


if __name__=="__main__":
    # for DAG recursive functions
    import sys
    print('recursion limit', sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    dataset = DAGDataSet(num_timeslots=1)

    greedy = HEFT(dataset=dataset)
    psoga = PSOGA(dataset=dataset, num_particles=100, w_max=0.2, w_min=0.05, c1=0.2, c2=0.1)
    genetic = Genetic(dataset=dataset, num_solutions=100, mutation_ratio=0.1, cross_over_ratio=0.1)

    start = time.time()
    x_lst, y_lst = greedy.run_algo()
    heft_result = result(x_lst, y_lst, took=time.time()-start, algorithm_name="Greedy Algorithm")

    dataset.system_manager.set_env(deployed_server=x_lst[0], execution_order=y_lst[0])
    for svc in dataset.system_manager.service_set.services:
        svc.deadline = np.mean(dataset.system_manager.total_time_dp())

    start = time.time()
    x_lst, y_lst = psoga.run_algo(loop=500, verbose=1, local_search=True)
    psoga_result = result(x_lst, y_lst, took=time.time()-start, algorithm_name="PSO-GA Algorithm")

    start = time.time()
    x_lst, y_lst = genetic.run_algo(loop=500, verbose=1, local_search=True)
    genetic_result = result(x_lst, y_lst, took=time.time()-start, algorithm_name="Genetic Algorithm")



    # coedge = True
    # if coedge:
    #     import config
    #     coedge_partition = dict()
    #     for i in range(7):
    #         coedge_partition[i] = []
    #     for l in config.service_info[0]['layers']:
    #         if l['layer_name'] == 'conv1':
    #             for i in range(6):
    #                 start = math.floor(math.floor(l['output_height'] / 4) / 6 * i)
    #                 end = math.floor(math.floor(l['output_height'] / 4) / 6 * (i + 1))
    #                 for j in range(start, end):
    #                     coedge_partition[i].append(l['layer_name']+'_'+str(j))
    #         elif l['layer_name'] == 'maxpool1' or l['layer_name'] == 'conv2' or l['layer_name'] == 'conv3':
    #             for i in range(6):
    #                 start = math.floor(math.floor(l['output_height'] / 2) / 6 * i)
    #                 end = math.floor(math.floor(l['output_height'] / 2) / 6 * (i + 1))
    #                 for j in range(start, end):
    #                     coedge_partition[i].append(l['layer_name']+'_'+str(j))
    #         elif l['layer_type'] == 'cnn' or l['layer_type'] == 'maxpool':
    #             for i in range(6):
    #                 start = math.floor(l['output_height'] / 6 * i)
    #                 end = math.floor(l['output_height'] / 6 * (i + 1))
    #                 for j in range(start, end):
    #                     coedge_partition[i].append(l['layer_name']+'_'+str(j))
    #         elif l['layer_type'] == 'fc':
    #             minimum_unit = 768
    #             start = 0
    #             end = math.floor(l['input_height'] * l['input_width'] * l['input_channel'] / 768)
    #             for j in range(start, end):
    #                 coedge_partition[6].append(l['layer_name']+'_'+str(j))

    #     for p_id in range(dataset.num_partitions):
    #         for i in range(7):
    #             if dataset.system_manager.service_set.partitions[p_id].layer_name in coedge_partition[i]:
    #                 dataset.coarsened_graph[p_id] = i

    # coedge_psoga = PSOGA(dataset=dataset, num_particles=100, w_max=0.2, w_min=0.05, c1=0.2, c2=0.1)
    # coedge_genetic = Genetic(dataset=dataset, num_solutions=100, mutation_ratio=0.1, cross_over_ratio=0.1)

    # start = time.time()
    # x_lst, y_lst = coedge_psoga.run_algo(loop=100, verbose=100, local_search=False)
    # coedge_psoga_result = result(x_lst, y_lst, took=time.time()-start, algorithm_name="CoEdge PSO-GA Algorithm")

    # start = time.time()
    # x_lst, y_lst = coedge_genetic.run_algo(loop=100, verbose=100, local_search=False)
    # coedge_genetic_result = result(x_lst, y_lst, took=time.time()-start, algorithm_name="CoEdge Genetic Algorithm")



    # energy_efficient = True
    # if energy_efficient:
    #     import config
    #     for idx, l in enumerate(config.service_info[0]['layers']):
    #         for p in dataset.system_manager.service_set.partitions:
    #             if l['layer_name'] == p.original_layer_name:
    #                 dataset.coarsened_graph[p.id] = idx

    # energy_efficient_psoga = PSOGA(dataset=dataset, num_particles=100, w_max=0.2, w_min=0.05, c1=0.2, c2=0.1)
    # energy_efficient_genetic = Genetic(dataset=dataset, num_solutions=100, mutation_ratio=0.1, cross_over_ratio=0.1)

    # start = time.time()
    # x_lst, y_lst = energy_efficient_psoga.run_algo(loop=100, verbose=100, local_search=False)
    # energy_efficient_psoga_result = result(x_lst, y_lst, took=time.time()-start, algorithm_name="Energy Efficient PSO-GA Algorithm")

    # start = time.time()
    # x_lst, y_lst = energy_efficient_genetic.run_algo(loop=100, verbose=100, local_search=False)
    # energy_efficient_genetic_result = result(x_lst, y_lst, took=time.time()-start, algorithm_name="Energy Efficient Genetic Algorithm")


    import matplotlib.pyplot as plt
