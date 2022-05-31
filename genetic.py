import numpy as np
import math
import random
import multiprocessing as mp
import time



class PSOGA:
    def __init__(self, dataset, num_particles, w_max=1.0, w_min=0.5, c1_s=0.5, c1_e=0.5, c2_s=0.5, c2_e=0.5):
        self.num_particles = num_particles
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots

        self.graph = [np.unique(cg) for cg in dataset.coarsened_graph]
        self.num_partitions = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])

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
        ps /= self.num_particles * self.num_timeslots * self.num_partitions
        w = self.w_max - (self.w_max - self.w_min) * np.exp(ps / (ps - 1.01))
        return w

    def generate_random_solutions(self, step=2):
        # return np.full(shape=(self.num_particles, self.num_timeslots, self.num_partitions), fill_value=self.server_lst[-1])
        # return np.array([[self.dataset.partition_device_map for _ in range(self.num_timeslots)] for _ in range(0, self.num_particles)])
        random_solutions = np.zeros((self.num_particles, self.num_timeslots, self.num_partitions))
        start = 0
        end = step
        if end > 0:
            random_solutions[start:end,:,:] = np.array([[self.dataset.partition_device_map for _ in range(self.num_timeslots)] for _ in range(start, end)])
        for i in self.server_lst:
            start = end
            end += step
            random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_partitions), fill_value=i)
        random_solutions[end:self.num_particles,:,:] = np.random.choice([0]+self.server_lst, size=(self.num_particles-end, self.num_timeslots, self.num_partitions))
        return random_solutions
    
    def local_search_multiprocessing(self, action):
        np.random.seed(random.randint(0,2147483647))
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(action[t])

            for j in np.random.choice(self.num_partitions, size=1, replace=False): # for jth layer
                # local search x: deployed_server
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]))
                max_reward = self.system_manager.get_reward()
                for s_id in [0]+self.server_lst:
                    if s_id == action[t,j]:
                        continue
                    if s_id == 0:
                        s_id = self.dataset.partition_device_map[j]
                    temp = action[t,j]
                    action[t,j] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]))
                    cur_reward = self.system_manager.get_reward()
                    if cur_reward > max_reward and all(self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(action[t]))):
                        max_reward = cur_reward
                    else:
                        action[t,j] = temp

            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(action[t]), timeslot=t)
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

    def run_algo(self, loop, verbose=True, local_search=True, random_solution=None):
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        if random_solution is None:
            x_t = self.generate_random_solutions()
        else:
            x_t = random_solution
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
            eval_lst.append(np.max(np.sum(p_t_eval_lst, axis=1)))

            if max_reward < np.max(np.sum(p_t_eval_lst, axis=1)):
                max_reward = np.max(np.sum(p_t_eval_lst, axis=1))
                not_changed_loop = 0
            elif not_changed_loop > 50:
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(g_t[0]))
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(g_t[0])
                print("t:", sum(t), t)
                print("e:", sum(e), e)
                print("r:", r, "\033[0m")
                return (np.array([self.get_uncoarsened_x(g_t[t]) for t in range(self.num_timeslots)]), eval_lst, time.time() - start)
            else:
                not_changed_loop += 1

            if i == loop - 1:
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(g_t[0]))
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(g_t[0])
                print("t:", sum(t), t)
                print("e:", sum(e), e, "\033[0m")

            # test
            end = time.time()
            if verbose and i % verbose == 0:
                print("---------- PSO-GA #{} loop ----------".format(i))
                self.system_manager.init_env()
                total_time = []
                total_energy = []
                total_reward = []
                for t in range(self.num_timeslots):
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(g_t[t]))
                    print("#timeslot {} x: {}".format(t, g_t[t]))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(self.system_manager.total_time_dp())
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(g_t[t]), timeslot=t)
                print("mean t: {:.5f}".format(np.sum(total_time, axis=None)), np.sum(np.array(total_time), axis=0) / self.dataset.num_timeslots)
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))
        return (np.array([self.get_uncoarsened_x(g_t[t]) for t in range(self.num_timeslots)]), eval_lst, time.time() - start)

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
        #     a_t[t,:self.num_partitions] = a_x
        #     b_t[t,:self.num_partitions] = b_x
        # if random.random() > 0.5:
        #     return a_t
        # else:
        #     return b_t
        np.random.seed(random.randint(0,2147483647))
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
            mutation_point = np.random.randint(low=0, high=self.num_partitions)

            # mutate x: deployed_server
            another_s_id = np.random.choice([0]+self.server_lst)
            x[mutation_point] = another_s_id
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
        
        temp = [self.mutation_multiprocessing(v_t[i]) for i in mutation_idx]
        v_t[mutation_idx] = np.array(temp)
        return v_t
        working_queue = [v_t[i] for i in mutation_idx]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.mutation_multiprocessing, working_queue))
        v_t[mutation_idx] = np.array(temp)
        return v_t

    # convert invalid action to valid action, deployed server action.
    def deployed_server_reparation(self, x):
        server_lst = [0] + self.server_lst
        cloud_lst = list(self.system_manager.cloud.keys())
        # request device constraint
        for c_id, s_id in enumerate(x):
            if s_id not in self.server_lst:
                x[c_id] = self.dataset.partition_device_map[c_id]
        while False in self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x)):
            # 각 서버에 대해서,
            for s_id in range(self.num_servers-1):
                deployed_container_lst = list(np.where(x == s_id)[0])
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                while self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x), s_id=s_id) == False:
                    # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        if another_s_id == 0:
                            another_s_id = self.dataset.partition_device_map[c_id]
                        if s_id != another_s_id and self.system_manager.server[another_s_id].cur_energy > 0 and self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x), s_id=another_s_id):
                            x[c_id] = another_s_id # 한번 넣어보고
                            if self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x), s_id=another_s_id): # 자원 넘치는지 확인.
                                break
                            else:
                                x[c_id] = s_id # 자원 넘치면 롤백

    # convert invalid action to valid action, multiprocessing function.
    def reparation_multiprocessing(self, postition):
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(postition[t])
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(postition[t]), timeslot=t)
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
            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(postition[t]))
            reward.append(self.system_manager.get_reward())
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(postition[t]), timeslot=t)
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
        self.num_partitions = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.mutation_ratio = mutation_ratio
        self.cross_over_ratio = cross_over_ratio

    def generate_random_solutions(self, step=2):
        # return np.full(shape=(self.num_solutions, self.num_timeslots, self.num_partitions), fill_value=self.server_lst[-1])
        # return np.array([[self.dataset.partition_device_map for _ in range(self.num_timeslots)] for _ in range(0, self.num_solutions)])
        random_solutions = np.zeros((self.num_solutions, self.num_timeslots, self.num_partitions))
        start = 0
        end = step
        if end > 0:
            random_solutions[start:end,:,:] = np.array([[self.dataset.partition_device_map for _ in range(self.num_timeslots)] for _ in range(start, end)])
        for i in self.server_lst:
            start = end
            end += step
            random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_partitions), fill_value=i)
        random_solutions[end:self.num_solutions,:,:] = np.random.choice([0]+self.server_lst, size=(self.num_solutions-end, self.num_timeslots, self.num_partitions))
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
        p_t = self.reparation(p_t)

        p_known = np.copy(p_t)
        if local_search:
            p_t = self.local_search(p_t, local_prob=0.2)

        for i in range(loop):
            q_t = self.selection(p_t, p_known)
            q_t = self.mutation(q_t, self.mutation_ratio)
            q_t = self.crossover(q_t, self.cross_over_ratio)
            q_t = self.reparation(q_t)

            if local_search:
                q_t = self.local_search(q_t, local_prob=0.2)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(p_t, q_t)
            eval_lst.append(v)

            if max_reward < v:
                max_reward = v
                not_changed_loop = 0
            elif not_changed_loop > 50:
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(p_t[0,0]))
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(p_t[0,0])
                print("t:", sum(t), t)
                print("e:", sum(e), e)
                print("r:", r, "\033[0m")
                return (np.array([self.get_uncoarsened_x(p_t[0,t]) for t in range(self.num_timeslots)]), eval_lst, time.time() - start)
            else:
                not_changed_loop += 1

            if i == loop - 1:
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(p_t[0,0]))
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                print(p_t[0,0])
                print("t:", sum(t), t)
                print("e:", sum(e), e, "\033[0m")

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
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(p_t[0,t]))
                    print("#timeslot {} x: {}".format(t, p_t[0,t]))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(self.system_manager.total_time_dp())
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(p_t[0,t]), timeslot=t)
                print("mean t: {:.5f}".format(np.sum(total_time, axis=None)), np.sum(np.array(total_time), axis=0) / self.dataset.num_timeslots)
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))
        return (np.array([self.get_uncoarsened_x(p_t[0,t]) for t in range(self.num_timeslots)]), eval_lst, time.time() - start)

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
            mutation_point = np.random.randint(low=0, high=self.num_partitions)

            # mutate x: deployed_server
            another_s_id = np.random.choice([0]+self.server_lst)
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

        self.graph = [cg for cg in dataset.coarsened_graph]
        self.num_partitions = sum([len(cg) for cg in dataset.coarsened_graph])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

    def run_algo(self):
        self.system_manager.rank_u = np.zeros(self.num_partitions)
        for svc in self.dataset.svc_set.services:
            for partition in svc.partitions:
                self.system_manager.calc_rank_u_average(partition)
        x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)
        y = np.array([np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_partitions)), reverse=False), dtype=np.int32)[:,1] for _ in range(self.num_timeslots)])

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
                    if self.system_manager.constraint_chk(deployed_server=x[t], execution_order=y[t], s_id=s_id):
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
            print("x[t]", x[t])
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


class Greedy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_timeslots = dataset.num_timeslots

        self.graph = [np.unique(cg) for cg in dataset.coarsened_graph]
        self.num_partitions = sum([len(np.unique(cg)) for cg in dataset.coarsened_graph])

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
        x = np.full(shape=(self.num_timeslots, self.num_partitions), fill_value=self.system_manager.cloud_id, dtype=np.int32)

        server_lst = self.server_lst

        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            for p_id in reversed(range(self.num_partitions)):
                # initialize the earliest finish time of the task
                minimum_latency = np.inf
                # for all available server, find earliest finish time server
                for s_id in [0] + self.server_lst:
                    if s_id == 0:
                        s_id = self.dataset.partition_device_map[p_id]
                    temp_x = x[t,p_id]
                    x[t,p_id] = s_id
                    if self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x[t]), s_id=s_id):
                        self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x[t]))
                        latency = sum(self.system_manager.total_time_dp())
                        if latency < minimum_latency:
                            minimum_latency = latency
                        else:
                            x[t,p_id] = temp_x
                    else:
                        x[t,p_id] = temp_x
            print("x[t]", x[t])
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(x[t]), timeslot=t)
        return np.array([self.get_uncoarsened_x(x[t]) for t in range(self.num_timeslots)])

        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            start = end = 0
            for svc in self.dataset.svc_set.services:
                start = end
                end += len(self.graph[svc.id])
                for p_id in reversed(self.graph[svc.id]):
                    # initialize the earliest finish time of the task
                    minimum_latency = np.inf
                    # for all available server, find earliest finish time server
                    for s_id in server_lst:
                        temp_x = x[t,start+p_id]
                        x[t,start+p_id] = s_id
                        if self.system_manager.constraint_chk(deployed_server=self.get_uncoarsened_x(x[t]), s_id=s_id):
                            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x[t]))
                            p_lst = [svc.partitions[real_p] for real_p in np.where(self.dataset.coarsened_graph[svc.id] == p_id)[0]]
                            self.system_manager.total_time_dp()
                            latency =  max([self.system_manager.finish_time[p.id] for p in p_lst])
                            # print(p_id,"latency",latency, s_id, self.system_manager.finish_time, [self.system_manager.finish_time[p.id] for p in p_lst])
                            # input()
                            if latency < minimum_latency:
                                minimum_latency = latency
                            else:
                                x[t,start+p_id] = temp_x
                        else:
                            x[t,start+p_id] = temp_x
                print("x[t]", x[t])
                input()
            print("x[t]", x[t])
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(x[t]), timeslot=t)
        return np.array([self.get_uncoarsened_x(x[t]) for t in range(self.num_timeslots)])