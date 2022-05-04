import numpy as np
import math
import random
import dag_completion_time
from dag_data_generator import DAGDataSet

import multiprocessing as mp
import time


class PSOGA:
    def __init__(self, dataset, num_particles, w_max=1.0, w_min=0.5, c1_s=0.5, c1_e=0.5, c2_s=0.5, c2_e=0.5):
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
        self.c1_s = c1_s                  # cognative constant (start)
        self.c1_e = c1_e                  # cognative constant (end)
        self.c2_s = c2_s                  # social constant (start)
        self.c2_e = c2_e                  # social constant (end)
    
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
        # random_solutions = np.zeros((self.num_particles, self.num_timeslots, self.num_partitions))
        # start = end = 0
        # for i in range(self.num_servers):
        #     if i == self.num_servers - 1:
        #         start = end
        #         end = self.num_particles
        #         random_solutions[start:end,:,:] = np.random.randint(low=0, high=self.num_servers-1, size=(end-start, self.num_timeslots, self.num_partitions))
        #     else:
        #         start = end
        #         end = start + 5
        #         random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_partitions), fill_value=i)
        # return random_solutions
        return np.random.randint(low=self.num_servers-2, high=self.num_servers-1, size=(self.num_particles, self.num_timeslots, self.num_partitions))
    
    def local_search_multiprocessing(self, action):
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(action[t])

            for j in np.random.choice(self.num_partitions, size=2, replace=False): # for jth layer
                # local search x: deployed_server
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst)
                max_reward = self.system_manager.get_reward()
                for s_id in range(self.num_servers-1):
                    if s_id == action[t,j]:
                        continue
                    temp = action[t,j]
                    action[t,j] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst)
                    cur_reward = self.system_manager.get_reward()
                    if cur_reward > max_reward and all(self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst)):
                        max_reward = cur_reward
                    else:
                        action[t,j] = temp

            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(action[t]), execution_order=self.scheduling_lst, timeslot=t)
        return action

    # we have to find local optimum from current chromosomes.
    def local_search(self, action, local_prob=0.2):
        local_idx = np.random.rand(action.shape[0])
        local_idx = local_idx < local_prob
        local_idx = np.where(local_idx)[0]
        if len(local_idx) == 0:
            local_idx = np.array([np.random.randint(low=0, high=action.shape[0])])
        np.random.shuffle(local_idx)

        working_queue = [action[i] for i in local_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.local_search_multiprocessing, working_queue))
        action[local_idx] = np.array(temp)
        return action

    def run_algo(self, loop, verbose=True, local_search=True):
        start = time.time()
        max_reward = 0
        not_changed_loop = 0

        x_t = self.generate_random_solutions()
        x_t = self.reparation(x_t)
        if local_search:
            x_t = self.local_search(x_t, local_prob=0.2)
        p_t, g_t, p_t_eval_lst = self.selection(x_t)

        for i in range(loop):
            v_t = self.mutation(x_t, g_t)
            c_t = self.crossover(v_t, p_t, crossover_rate=(self.c1_e-self.c1_s) * (i / loop) + self.c1_s)
            x_t = self.crossover(c_t, g_t, crossover_rate=(self.c2_e-self.c2_s) * (i / loop) + self.c2_s, is_global=True)
            x_t = self.reparation(x_t)
            
            if local_search:
                x_t = self.local_search(x_t, local_prob=0.2)
            p_t, g_t, p_t_eval_lst = self.selection(x_t, p_t, g_t, p_t_eval_lst=p_t_eval_lst)

            if max_reward < np.max(np.sum(p_t_eval_lst, axis=1)):
                max_reward = np.max(np.sum(p_t_eval_lst, axis=1))
                not_changed_loop = 0
            elif not_changed_loop > 100:
                print("\033[31mEarly exit loop {}: {:.5f} sec\033[0m".format(i, time.time() - start))
                return np.array([self.get_uncoarsened_x(g_t[t]) for t in range(self.num_timeslots)]), np.array([self.scheduling_lst for t in range(self.num_timeslots)])
            else:
                not_changed_loop += 1

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                print("---------- PSO-GA #{} loop ----------".format(i))
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
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(g_t[t]), execution_order=self.scheduling_lst, timeslot=t)
                print("mean t: {:.5f}".format(sum(total_time) / self.num_timeslots))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))
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
        if len(crossover_idx) == 0:
            crossover_idx = np.array([np.random.randint(low=0, high=self.num_particles)])
        
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
        if len(mutation_idx) == 0:
            mutation_idx = np.array([np.random.randint(low=0, high=self.num_particles)])
        
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


class PSOGA_XY(PSOGA):
    def generate_random_solutions(self):
        solutions = []
        for _ in range(self.num_particles):
            x = np.random.randint(low=self.num_servers-2, high=self.num_servers-1, size=(self.num_timeslots, self.num_partitions))
            y = []
            for _ in range(self.num_timeslots):
                y.append(self.scheduling_lst)
            solutions.append(np.concatenate([x, np.array(y)], axis=-1))
        solutions = np.array(solutions)
        return solutions
    
    def local_search_multiprocessing(self, action):
        x_lst = np.array(action[:,:self.num_partitions])
        y_lst = np.array(action[:,self.num_partitions:])
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(x_lst[t], y_lst[t])

            for j in np.random.choice(self.num_partitions, size=1, replace=False): # for jth layer
                # local search x: deployed_server
                self.system_manager.set_env(deployed_server=x_lst[t], execution_order=y_lst[t])
                max_reward = self.system_manager.get_reward()
                for s_id in range(self.num_servers-1):
                    if s_id == x_lst[t,j]:
                        continue
                    temp = x_lst[t,j]
                    x_lst[t,j] = s_id
                    self.system_manager.set_env(deployed_server=x_lst[t], execution_order=y_lst[t])
                    cur_reward = self.system_manager.get_reward()
                    if cur_reward > max_reward and all(self.system_manager.constraint_chk(deployed_server=x_lst[t], execution_order=y_lst[t])):
                        max_reward = cur_reward
                    else:
                        x_lst[t,j] = temp

            for j in np.random.choice(self.num_partitions, size=1, replace=False): # for jth layer
                # local search y: execution order
                for p_id in range(self.num_partitions):
                    if p_id == j or self.system_manager.service_set.partitions[j].service.total_predecessors[p_id, j] or self.system_manager.service_set.partitions[j].service.total_predecessors[j, p_id]:
                        continue
                    temp = y_lst[t,j]
                    y_lst[t,j] = y_lst[t,p_id]
                    y_lst[t,p_id] = temp
                    self.system_manager.set_env(deployed_server=x_lst[t], execution_order=y_lst[t])
                    cur_reward = self.system_manager.get_reward()
                    if cur_reward > max_reward and all(self.system_manager.constraint_chk(deployed_server=x_lst[t], execution_order=y_lst[t])):
                        max_reward = cur_reward
                    else:
                        temp = y_lst[t,j]
                        y_lst[t,j] = y_lst[t,p_id]
                        y_lst[t,p_id] = temp

            self.system_manager.after_timeslot(deployed_server=x_lst[t], execution_order=y_lst[t], timeslot=t)
        action = np.concatenate([x_lst, y_lst], axis=-1)
        return action

    def run_algo(self, loop, verbose=True, local_search=True):
        start = time.time()
        max_reward = 0
        not_changed_loop = 0

        x_t = self.generate_random_solutions()
        x_t = self.reparation(x_t)
        if local_search:
            x_t = self.local_search(x_t, local_prob=0.2)
        p_t, g_t, p_t_eval_lst = self.selection(x_t)

        for i in range(loop):
            v_t = self.mutation(x_t, g_t)
            c_t = self.crossover(v_t, p_t, crossover_rate=self.c1)
            x_t = self.crossover(c_t, g_t, crossover_rate=self.c2, is_global=True)
            x_t = self.reparation(x_t)
            
            if local_search:
                x_t = self.local_search(x_t, local_prob=0.2)
            p_t, g_t, p_t_eval_lst = self.selection(x_t, p_t, g_t, p_t_eval_lst=p_t_eval_lst)

            if max_reward < np.max(np.sum(p_t_eval_lst, axis=1)):
                max_reward = np.max(np.sum(p_t_eval_lst, axis=1))
                not_changed_loop = 0
            elif not_changed_loop > 100:
                print("\033[31mEarly exit loop {}: {:.5f} sec\033[0m".format(i, time.time() - start))
                return g_t[:,:self.num_partitions], g_t[:,self.num_partitions:]
            else:
                not_changed_loop += 1

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                print("---------- PSO-GA XY #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    x = np.array(g_t[t,:self.num_partitions])
                    y = np.array(g_t[t,self.num_partitions:])
                    self.system_manager.set_env(deployed_server=x, execution_order=y)
                    print("#timeslot {} x: {}".format(t, x))
                    print("#timeslot {} y: {}".format(t, y))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(sum(self.system_manager.total_time_dp()))
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=x, execution_order=y, timeslot=t)
                print("mean t: {:.5f}".format(sum(total_time) / self.num_timeslots))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))
        return g_t[:,:self.num_partitions], g_t[:,self.num_partitions:]

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
            a_x = a_t[t,:self.num_partitions]
            a_y = a_t[t,self.num_partitions:]
            b_x = b_t[t,:self.num_partitions]
            b_y = b_t[t,self.num_partitions:]

            cross_point = np.random.randint(low=0, high=a_x.size, size=2)

            # crossover x: deployed_server
            a_x[cross_point[0]:cross_point[1]] = b_x[cross_point[0]:cross_point[1]]
            a_t[t,:self.num_partitions] = a_x

            # crossover y: execution_order
            for j in range(cross_point[0], cross_point[1]):
                for k in range(self.num_partitions):
                    if b_y[j] == a_y[k]:
                        temp = a_y[j]
                        a_y[j] = a_y[k]
                        a_y[k] = temp
                        break
            a_t[t,self.num_partitions:] = a_y
        return a_t

    def mutation_multiprocessing(self, v_t):
        for t in range(self.num_timeslots):
            x = v_t[t,:self.num_partitions]
            y = v_t[t,self.num_partitions:]

            mutation_point = np.random.randint(low=0, high=self.num_partitions)

            # mutate x: deployed_server
            x[mutation_point] = np.random.randint(low=0, high=self.num_servers-1)
            v_t[t,:self.num_partitions] = x

            # mutate y: execution_order
            rand = np.random.randint(low=0, high=self.num_partitions)
            for k in range(self.num_partitions):
                if rand == y[k]:
                    temp = y[mutation_point]
                    y[mutation_point] = y[k]
                    y[k] = temp
                    break
            v_t[t,self.num_partitions:] = y
        return v_t

    # convert invalid action to valid action, deployed server action.
    def deployed_server_reparation(self, x, y):
        server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())
        cloud_lst = list(self.system_manager.cloud.keys())
        while False in self.system_manager.constraint_chk(deployed_server=x, execution_order=y):
            # 각 서버에 대해서,
            for s_id in range(self.num_servers-1):
                deployed_container_lst = list(np.where(x == s_id)[0])
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                while self.system_manager.constraint_chk(deployed_server=x, execution_order=y, s_id=s_id) == False:
                    # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        if s_id != another_s_id and self.system_manager.server[another_s_id].cur_energy > 0 and self.system_manager.constraint_chk(deployed_server=x, execution_order=y, s_id=another_s_id):
                            x[c_id] = another_s_id # 한번 넣어보고
                            if self.system_manager.constraint_chk(deployed_server=x, execution_order=y, s_id=another_s_id): # 자원 넘치는지 확인.
                                break
                            else:
                                x[c_id] = s_id # 자원 넘치면 롤백

    # convert invalid action to valid action, multiprocessing function.
    def reparation_multiprocessing(self, postition):
        x_lst = np.array(postition[:,:self.num_partitions])
        y_lst = np.array(postition[:,self.num_partitions:])
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(x_lst[t], y_lst[t])
            self.system_manager.after_timeslot(deployed_server=x_lst[t], execution_order=y_lst[t], timeslot=t)
        postition = np.concatenate([x_lst, y_lst], axis=-1)
        return postition
    
    def evaluation_multiprocessing(self, postition):
        x_lst = np.array(postition[:,:self.num_partitions])
        y_lst = np.array(postition[:,self.num_partitions:])
        self.system_manager.init_env()
        reward = []
        for t in range(self.num_timeslots):
            self.system_manager.set_env(deployed_server=x_lst[t], execution_order=y_lst[t])
            reward.append(self.system_manager.get_reward())
            self.system_manager.after_timeslot(deployed_server=x_lst[t], execution_order=y_lst[t], timeslot=t)
        return reward


class Genetic(PSOGA):
    def __init__(self, dataset, num_solutions, mutation_ratio=0.5, cross_over_ratio=0.7):
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
        # random_solutions = np.zeros((self.num_solutions, self.num_timeslots, self.num_partitions))
        # start = end = 0
        # for i in range(self.num_servers):
        #     if i == self.num_servers - 1:
        #         start = end
        #         end = self.num_solutions
        #         random_solutions[start:end,:,:] = np.random.randint(low=0, high=self.num_servers-1, size=(end-start, self.num_timeslots, self.num_partitions))
        #     else:
        #         start = end
        #         end = start + 5
        #         random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_partitions), fill_value=i)
        # return random_solutions
        return np.random.randint(low=self.num_servers-2, high=self.num_servers-1, size=(self.num_solutions, self.num_timeslots, self.num_partitions))

    def run_algo(self, loop, verbose=True, local_search=True):
        start = time.time()
        max_reward = 0
        not_changed_loop = 0

        p_t = self.generate_random_solutions()
        p_t = self.reparation(p_t)

        p_known = np.copy(p_t)
        if local_search:
            p_t = self.local_search(p_t, local_prob=0.2)

        for i in range(loop):
            q_t = self.selection(p_t, p_known)

            q_t = self.crossover(q_t, self.cross_over_ratio)
            q_t = self.mutation(q_t, self.mutation_ratio)
            q_t = self.reparation(q_t)

            if local_search:
                q_t = self.local_search(q_t, local_prob=0.2)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(p_t, q_t)

            if max_reward < v:
                max_reward = v
                not_changed_loop = 0
            elif not_changed_loop > 100:
                print("\033[31mEarly exit loop {}: {:.5f} sec\033[0m".format(i, time.time() - start))
                return np.array([self.get_uncoarsened_x(p_t[0,t]) for t in range(self.num_timeslots)]), np.array([self.scheduling_lst for t in range(self.num_timeslots)])
            else:
                not_changed_loop += 1

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                print("---------- Genetic #{} loop ----------".format(i))
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
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(p_t[0,t]), execution_order=self.scheduling_lst, timeslot=t)
                print("mean t: {:.5f}".format(sum(total_time) / self.num_timeslots))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))
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
        ev_lst = np.sum(self.evaluation(union), axis=1)
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
        if len(crossover_idx) < 2:
            crossover_idx = np.array([np.random.randint(low=0, high=self.num_solutions, size=2)])
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
        if len(mutation_idx) == 0:
            mutation_idx = np.array([np.random.randint(low=0, high=self.num_solutions)])
        
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
            Tr = np.zeros(self.num_partitions)
            Tf = np.zeros(self.num_partitions)
            for top_rank in y[t]:
                # initialize the earliest finish time of the task
                earliest_finish_time = np.inf
                # for all available server, find earliest finish time server
                for s_id in server_lst:
                    temp_x = x[t,top_rank]
                    x[t,top_rank] = s_id
                    if self.system_manager.constraint_chk(x[t], y[t], s_id):
                        self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                        finish_time, _, _ = self.system_manager.get_completion_time(top_rank, np.copy(Tr), np.copy(Tf))
                        if finish_time < earliest_finish_time:
                            earliest_finish_time = finish_time
                        else:
                            x[t,top_rank] = temp_x
                    else:
                        x[t,top_rank] = temp_x
                self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                finish_time, Tr, Tf = self.system_manager.get_completion_time(top_rank, np.copy(Tr), np.copy(Tf))
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)

    def run_algo_layer(self, dataset):
        coarsened_graph = np.zeros_like(dataset.coarsened_graph)
        for idx, l in enumerate(dataset.service_info[0]['layers']):
            for p in dataset.system_manager.service_set.partitions:
                if l['layer_name'] == p.original_layer_name:
                    coarsened_graph[p.id] = idx
        
        x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        y = np.array([np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(self.num_timeslots)])

        layer_graph = np.unique(coarsened_graph)
        num_layers = len(layer_graph)
        layer_x = np.full(shape=(self.num_timeslots, num_layers), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        layer_y = np.array([np.zeros_like(layer_graph) for _ in range(self.num_timeslots)])
        for t in range(self.num_timeslots):
            rank_u = np.zeros(num_layers)
            for k in layer_graph:
                rank_u[k] = max(self.system_manager.rank_u[np.where(coarsened_graph == k)])
            layer_y[t] = np.array(sorted(zip(rank_u, layer_graph), reverse=True), dtype=np.int32)[:,1]

        server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            for top_rank in layer_y[t]:
                # initialize the earliest finish time of the task
                earliest_finish_time = np.inf
                # for all available server, find earliest finish time server
                for s_id in server_lst:
                    temp_x = layer_x[t,top_rank]
                    layer_x[t,top_rank] = s_id
                    x[t,np.where(coarsened_graph == top_rank)] = s_id
                    if self.system_manager.constraint_chk(x[t], y[t], s_id):
                        self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
                        self.system_manager.total_time_dp()
                        finish_time = max(self.system_manager.service_set.services[0].finish_time[np.where(coarsened_graph == top_rank)])
                        if finish_time < earliest_finish_time:
                            earliest_finish_time = finish_time
                        else:
                            layer_x[t,top_rank] = temp_x
                            x[t,np.where(coarsened_graph == top_rank)] = temp_x
                    else:
                        layer_x[t,top_rank] = temp_x
                        x[t,np.where(coarsened_graph == top_rank)] = temp_x
            
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


def result(dataset, x_lst, y_lst, took, algorithm_name):
    total_time = []
    total_energy = []
    total_reward = []
    print("\033[95m---------- " + algorithm_name + " Test! ----------\033[96m")
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
        total_energy.append(sum([s.energy_consumption() for s in list(dataset.system_manager.local.values()) + list(dataset.system_manager.edge.values())]))
        total_reward.append(dataset.system_manager.get_reward())
        dataset.system_manager.after_timeslot(deployed_server=x, execution_order=y, timeslot=t)
    print("mean t: {:.5f}".format(sum(total_time) / dataset.num_timeslots))
    print("mean e: {:.5f}".format(sum(total_energy) / dataset.num_timeslots))
    print("mean r: {:.5f}".format(sum(total_reward) / dataset.num_timeslots))
    print("took: {:.5f} sec".format(took))
    print("\033[95m---------- " + algorithm_name + " Done! ----------\033[0m\n")
    return total_time, total_energy, total_reward


if __name__=="__main__":
    # for DAG recursive functions
    import sys
    print('recursion limit', sys.getrecursionlimit())
    sys.setrecursionlimit(10000)

    test_dataset = DAGDataSet(num_timeslots=1, num_partitions=[50,50,50,10], apply_partition=False)

    layer_greedy = HEFT(dataset=test_dataset)
    layer_psoga_xy = PSOGA_XY(dataset=test_dataset, num_particles=100, w_max=0.8, w_min=0.2, c1_s=0.8, c1_e=0.2, c2_s=0.4, c2_e=0.8)
    layer_psoga = PSOGA(dataset=test_dataset, num_particles=100, w_max=0.8, w_min=0.2, c1_s=0.8, c1_e=0.2, c2_s=0.4, c2_e=0.8)
    layer_genetic = Genetic(dataset=test_dataset, num_solutions=100, mutation_ratio=0.2, cross_over_ratio=0.4)

    start = time.time()
    x_lst = np.full(shape=(test_dataset.num_timeslots, test_dataset.num_partitions), fill_value=test_dataset.system_manager.net_manager.request_device, dtype=np.int32)
    y_lst = np.array([np.array(sorted(zip(test_dataset.system_manager.rank_u, np.arange(test_dataset.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(test_dataset.num_timeslots)])
    edge_result = result(test_dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Test Local Only")
    print("get_computation_time", sum([p.get_computation_time() for p in test_dataset.system_manager.service_set.partitions]))

    start = time.time()
    x_lst = np.full(shape=(test_dataset.num_timeslots, test_dataset.num_partitions), fill_value=list(test_dataset.system_manager.edge.keys())[0], dtype=np.int32)
    y_lst = np.array([np.array(sorted(zip(test_dataset.system_manager.rank_u, np.arange(test_dataset.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(test_dataset.num_timeslots)])
    edge_result = result(test_dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Test Edge Only")
    print("get_computation_time", sum([p.get_computation_time() for p in test_dataset.system_manager.service_set.partitions]))

    # start = time.time()
    # x_lst, y_lst = layer_greedy.run_algo()
    # heft_result = result(test_dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Layer-based HEFT Algorithm")

    # test_dataset.system_manager.set_env(deployed_server=x_lst[0], execution_order=y_lst[0])
    # delay = test_dataset.system_manager.total_time_dp()
    # for svc in test_dataset.system_manager.service_set.services:
    #     svc.deadline = delay[svc.id] * 0.5

    # start = time.time()
    # x_lst, y_lst = layer_psoga_xy.run_algo(loop=500, verbose=100, local_search=False)
    # psoga_xy_result = result(test_dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="PSO-GA_XY Algorithm")

    start = time.time()
    x_lst, y_lst = layer_psoga.run_algo(loop=500, verbose=100, local_search=False)
    psoga_result = result(test_dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="PSO-GA Algorithm")

    start = time.time()
    x_lst, y_lst = layer_genetic.run_algo(loop=500, verbose=100, local_search=False)
    genetic_result = result(test_dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Genetic Algorithm")



    # print("\n========== Our Scheme Start ==========\n")

    # dataset = DAGDataSet(num_timeslots=1, num_partitions=[50,50,50,10], apply_partition=False, net_manager=test_dataset.system_manager.net_manager, svc_arrival=test_dataset.system_manager.service_arrival)
    # #dataset.system_manager.scheduling_policy = 'EFT'

    # greedy = HEFT(dataset=dataset)
    # psoga = PSOGA(dataset=dataset, num_particles=100, w_max=0.8, w_min=0.2, c1_s=0.8, c1_e=0.2, c2_s=0.4, c2_e=0.8)
    # genetic = Genetic(dataset=dataset, num_solutions=100, mutation_ratio=0.2, cross_over_ratio=0.4)

    # start = time.time()
    # x_lst = np.full(shape=(dataset.num_timeslots, dataset.num_partitions), fill_value=dataset.system_manager.net_manager.request_device, dtype=np.int32)
    # y_lst = np.array([np.array(sorted(zip(dataset.system_manager.rank_u, np.arange(dataset.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(dataset.num_timeslots)])
    # edge_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Local Only")
    # print("get_computation_time", sum([p.get_computation_time() for p in dataset.system_manager.service_set.partitions]))

    # start = time.time()
    # x_lst = np.full(shape=(dataset.num_timeslots, dataset.num_partitions), fill_value=list(dataset.system_manager.edge.keys())[0], dtype=np.int32)
    # y_lst = np.array([np.array(sorted(zip(dataset.system_manager.rank_u, np.arange(dataset.num_partitions)), reverse=True), dtype=np.int32)[:,1] for _ in range(dataset.num_timeslots)])
    # edge_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Edge Only")
    # print("get_computation_time", sum([p.get_computation_time() for p in dataset.system_manager.service_set.partitions]))

    # # start = time.time()
    # # x_lst, y_lst = greedy.run_algo_layer(dataset)
    # # heft_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Partition-based HEFT Algorithm")

    # # dataset.system_manager.set_env(deployed_server=x_lst[0], execution_order=y_lst[0])
    # # delay = dataset.system_manager.total_time_dp()
    # # for svc in dataset.system_manager.service_set.services:
    # #     svc.deadline = delay[svc.id] * 0.5

    # # start = time.time()
    # # x_lst, y_lst = psoga.run_algo(loop=500, verbose=100, local_search=False)
    # # psoga_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="PSO-GA Algorithm")

    # # start = time.time()
    # # x_lst, y_lst = genetic.run_algo(loop=500, verbose=100, local_search=True)
    # # genetic_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Genetic Algorithm")



    # print("\n========== CoEdge Start ==========\n")
    # coedge = True
    # if coedge:
    #     coedge_partition = dict()
    #     if dataset.service_info[0]['model_name'] == 'GoogLeNet':
    #         for i in range(7):
    #             coedge_partition[i] = []
    #         for l in dataset.service_info[0]['layers']:
    #             p_lst = [p.layer_name for p in dataset.system_manager.service_set.partitions if p.original_layer_name == l['layer_name']]
    #             if l['layer_type'] == 'cnn' or l['layer_type'] == 'maxpool':
    #                 for i in range(6):
    #                     start = math.floor(len(p_lst) / 6 * i)
    #                     end = math.floor(len(p_lst) / 6 * (i + 1))
    #                     for j in range(start, end):
    #                         coedge_partition[i].append(l['layer_name']+'_'+str(j))
    #             elif l['layer_type'] == 'fc':
    #                 coedge_partition[6].extend(p_lst)

    #         for p_id in range(dataset.num_partitions):
    #             for i in range(7):
    #                 if dataset.system_manager.service_set.partitions[p_id].layer_name in coedge_partition[i]:
    #                     dataset.coarsened_graph[p_id] = i

    #     elif dataset.service_info[0]['model_name'] == 'AlexNet':
    #         for i in range(7):
    #             coedge_partition[i] = []
    #         for l in dataset.service_info[0]['layers']:
    #             p_lst = [p.layer_name for p in dataset.system_manager.service_set.partitions if p.original_layer_name == l['layer_name']]
    #             if l['layer_type'] == 'cnn' or l['layer_type'] == 'maxpool':
    #                 for i in range(6):
    #                     start = math.floor(len(p_lst) / 6 * i)
    #                     end = math.floor(len(p_lst) / 6 * (i + 1))
    #                     for j in range(start, end):
    #                         coedge_partition[i].append(l['layer_name']+'_'+str(j))
    #             elif l['layer_type'] == 'fc':
    #                 coedge_partition[6].extend(p_lst)

    #         for p_id in range(dataset.num_partitions):
    #             for i in range(7):
    #                 if dataset.system_manager.service_set.partitions[p_id].layer_name in coedge_partition[i]:
    #                     dataset.coarsened_graph[p_id] = i

    #     elif dataset.service_info[0]['model_name'] == 'ResNet-50':
    #         for i in range(8):
    #             coedge_partition[i] = []
    #         for l in dataset.service_info[0]['layers']:
    #             p_lst = [p.layer_name for p in dataset.system_manager.service_set.partitions if p.original_layer_name == l['layer_name']]
    #             if l['layer_type'] == 'cnn' or l['layer_type'] == 'maxpool' or l['layer_type'] == 'avgpool':
    #                 for i in range(7):
    #                     start = math.floor(len(p_lst) / 7 * i)
    #                     end = math.floor(len(p_lst) / 7 * (i + 1))
    #                     for j in range(start, end):
    #                         coedge_partition[i].append(l['layer_name']+'_'+str(j))
    #             elif l['layer_type'] == 'fc':
    #                 coedge_partition[7].extend(p_lst)

    #         for p_id in range(dataset.num_partitions):
    #             for i in range(8):
    #                 if dataset.system_manager.service_set.partitions[p_id].layer_name in coedge_partition[i]:
    #                     dataset.coarsened_graph[p_id] = i

    #     coedge_psoga = PSOGA(dataset=dataset, num_particles=100, w_max=0.8, w_min=0.2, c1_s=0.8, c1_e=0.2, c2_s=0.4, c2_e=0.8)
    #     coedge_genetic = Genetic(dataset=dataset, num_solutions=100, mutation_ratio=0.2, cross_over_ratio=0.4)

    #     start = time.time()
    #     x_lst, y_lst = coedge_psoga.run_algo(loop=500, verbose=100, local_search=False)
    #     coedge_psoga_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="CoEdge PSO-GA Algorithm")

    #     start = time.time()
    #     x_lst, y_lst = coedge_genetic.run_algo(loop=500, verbose=100, local_search=True)
    #     coedge_genetic_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="CoEdge Genetic Algorithm")



    # energy_efficient = True
    # if energy_efficient:
    #     for idx, l in enumerate(dataset.service_info[0]['layers']):
    #         for p in dataset.system_manager.service_set.partitions:
    #             if l['layer_name'] == p.original_layer_name:
    #                 dataset.coarsened_graph[p.id] = idx

    #     energy_efficient_psoga = PSOGA(dataset=dataset, num_particles=100, w_max=0.8, w_min=0.2, c1_s=0.8, c1_e=0.2, c2_s=0.4, c2_e=0.8)
    #     energy_efficient_genetic = Genetic(dataset=dataset, num_solutions=100, mutation_ratio=0.2, cross_over_ratio=0.4)

    #     start = time.time()
    #     x_lst, y_lst = energy_efficient_psoga.run_algo(loop=500, verbose=100, local_search=False)
    #     energy_efficient_psoga_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Energy Efficient PSO-GA Algorithm")

    #     start = time.time()
    #     x_lst, y_lst = energy_efficient_genetic.run_algo(loop=500, verbose=100, local_search=True)
    #     energy_efficient_genetic_result = result(dataset, x_lst, y_lst, took=time.time()-start, algorithm_name="Energy Efficient Genetic Algorithm")


    import matplotlib.pyplot as plt
