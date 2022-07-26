from importlib.abc import ExecutionLoader
from re import X
from matplotlib.pyplot import xcorr
import numpy as np
import math
import random
import multiprocessing as mp
import time
from copy import deepcopy

from torch import minimum



class Layerwise_PSOGA:
    def __init__(self, dataset, num_particles, w_max=0.8, w_min=0.2, c1_s=0.9, c1_e=0.2, c2_s=0.4, c2_e=0.9):
        self.num_particles = num_particles
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_partitions = dataset.num_partitions
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.w_max = w_max            # constant inertia max weight (how much to weigh the previous velocity)
        self.w_min = w_min            # constant inertia min weight (how much to weigh the previous velocity)
        self.c1_s = c1_s                  # cognative constant (start)
        self.c1_e = c1_e                  # cognative constant (end)
        self.c2_s = c2_s                  # social constant (start)
        self.c2_e = c2_e                  # social constant (end)

    def update_w(self, x_t, g_t):
        ps = 0
        for i in range(self.num_particles):
            ps += np.sum(np.equal(x_t[i], g_t), axis=None)
        ps /= self.num_particles * self.num_timeslots * self.num_partitions
        w = self.w_max - (self.w_max - self.w_min) * np.exp(ps / (ps - 1.01))
        return w

    def generate_random_solutions(self, greedy_solution=None, step=0):
        # if greedy_solution is not None:
        #     action = []
        #     for idx, sol in enumerate(greedy_solution):
        #         action.extend([greedy_solution[idx] for _ in range(idx * math.ceil(self.num_particles/len(greedy_solution)), (idx+1)*math.ceil(self.num_particles/len(greedy_solution)))])
        #     return np.array(action)
        # server = np.zeros((self.num_particles, self.num_timeslots, self.num_partitions))
        # start = end = 0
        # for i in self.server_lst:
        #     start = end
        #     end += step
        #     server[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_partitions), fill_value=i)
        # server[end:self.num_particles,:,:] = np.random.choice(self.server_lst, size=(self.num_particles-end, self.num_timeslots, self.num_partitions))
        server = np.random.choice(self.server_lst, size=(self.num_particles, self.num_timeslots, self.num_partitions))
        order = np.array([np.random.choice(self.num_partitions, size=self.num_partitions, replace=False) for _ in range(self.num_particles)]).reshape(self.num_particles, self.num_timeslots, self.num_partitions) # self.system_manager.rank_u_schedule
        action = np.concatenate([server, order], axis=2)
        return action
    
    def local_search_multiprocessing(self, action):
        np.random.seed(random.randint(0,2147483647))
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            # self.deployed_server_reparation(action[t])
            x = action[t,:self.num_partitions]
            y = action[t,self.num_partitions:]

            self.system_manager.set_env(deployed_server=x, execution_order=y)
            min_total_time = self.system_manager.total_time_dp()
            for j in np.random.choice(self.num_partitions, size=2, replace=False): # math.ceil(self.num_partitions/10)
                # local search x: deployed_server
                for s_id in self.server_lst:
                    if s_id == 0:
                        s_id = self.dataset.partition_device_map[j]
                    temp = x[j]
                    x[j] = s_id
                    self.system_manager.set_env(deployed_server=x, execution_order=y)
                    cur_total_time = self.system_manager.total_time_dp()
                    if max(cur_total_time) < max(min_total_time) and all(self.system_manager.constraint_chk()):
                        min_total_time = cur_total_time
                    else:
                        x[j] = temp
                # local search y: execution_order
                for order in np.random.choice(max(round(y[j]), 2), size=2, replace=False):
                    temp = y[j]
                    y[j] = order
                    self.system_manager.set_env(deployed_server=x, execution_order=y)
                    cur_total_time = self.system_manager.total_time_dp()
                    if max(cur_total_time) < max(min_total_time) and all(self.system_manager.constraint_chk()):
                        min_total_time = cur_total_time
                    else:
                        y[j] = temp

            self.system_manager.after_timeslot(deployed_server=x, execution_order=y, timeslot=t)
            action[t] = np.concatenate([x, y], axis=0)
        return action

    # we have to find local optimum from current chromosomes.
    def local_search(self, action, local_prob=0.5):
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

    def run_algo(self, loop, verbose=True, local_search=False, greedy_solution=None, early_exit_loop=50):
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        x_t = self.generate_random_solutions(greedy_solution)
        x_t = self.reparation(x_t)
        if local_search:
            x_t = self.local_search(x_t, local_prob=0.5)
        p_t, g_t, p_t_eval_lst = self.selection(x_t)

        for i in range(loop):
            v_t = self.mutation(x_t, g_t)
            c_t = self.crossover(v_t, p_t, crossover_rate=(self.c1_e-self.c1_s) * (i / loop) + self.c1_s)
            x_t = self.crossover(c_t, g_t, crossover_rate=(self.c2_e-self.c2_s) * (i / loop) + self.c2_s, is_global=True)
            x_t = self.reparation(x_t)
            if local_search:
                x_t = self.local_search(x_t, local_prob=0.5)
            
            p_t, g_t, p_t_eval_lst = self.selection(x_t, p_t, g_t, p_t_eval_lst=p_t_eval_lst)
            eval_lst.append(np.max(np.sum(p_t_eval_lst, axis=1)))

            if max_reward < np.max(np.sum(p_t_eval_lst, axis=1)):
                max_reward = np.max(np.sum(p_t_eval_lst, axis=1))
                not_changed_loop = 0
            elif not_changed_loop > early_exit_loop:
                self.system_manager.set_env(deployed_server=g_t[0,:self.num_partitions], execution_order=g_t[0,self.num_partitions:])
                t = self.system_manager.total_time_dp()
                e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
                r = self.system_manager.get_reward()
                print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
                # print(g_t[0])
                print("t:", max(t), t)
                print("e:", sum(e), e)
                print("r:", r, "\033[0m")
                return ((g_t[:,:self.num_partitions], g_t[:,self.num_partitions:]), eval_lst, time.time() - start)
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
                    self.system_manager.set_env(deployed_server=g_t[t,:self.num_partitions], execution_order=g_t[t,self.num_partitions:])
                    #print("#timeslot {} x: {}".format(t, g_t[t]))
                    print("#timeslot {} constraint: {}".format(t, [s.constraint_chk() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} m: {}".format(t, [(s.memory - max(s.deployed_partition_memory.values(), default=0)) / s.memory for s in self.system_manager.server.values()]))
                    #print("#timeslot {} e: {}".format(t, [s.cur_energy - s.energy_consumption() for s in self.system_manager.server.values()]))
                    #print("#timeslot {} t: {}".format(t, self.system_manager.total_time_dp()))
                    total_time.append(self.system_manager.total_time_dp())
                    total_energy.append(sum([s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]))
                    total_reward.append(self.system_manager.get_reward())
                    self.system_manager.after_timeslot(deployed_server=g_t[t,:self.num_partitions], execution_order=g_t[t,self.num_partitions:], timeslot=t)
                print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))

        self.system_manager.set_env(deployed_server=g_t[0,:self.num_partitions], execution_order=g_t[0,self.num_partitions:])
        t = self.system_manager.total_time_dp()
        e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
        r = self.system_manager.get_reward()
        print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
        # print(g_t[0])
        print("t:", max(t), t)
        print("e:", sum(e), e)
        print("r:", r, "\033[0m")
        return ((g_t[:,:self.num_partitions], g_t[:,self.num_partitions:]), eval_lst, time.time() - start)

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
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t = inputs
        for t in range(self.num_timeslots):
            a_x = a_t[t,:self.num_partitions]
            a_y = a_t[t,self.num_partitions:]
            b_x = b_t[t,:self.num_partitions]
            b_y = b_t[t,self.num_partitions:]
            cross_point = np.random.randint(low=0, high=a_x.size, size=2) # math.ceil(self.num_partitions/10)

            # crossover x: deployed_server
            for i in cross_point:
                a_x[i] = b_x[i]
            a_t[t,:self.num_partitions] = a_x

            # crossover y: execution_order
            for j in cross_point:
                for k in range(self.num_partitions):
                    if b_y[j] == a_y[k]:
                        temp = a_y[j]
                        a_y[j] = a_y[k]
                        a_y[k] = temp
                        break
            a_t[t,self.num_partitions:] = a_y

            # # crossover x: deployed_server
            # a_x[cross_point[0]:cross_point[1]] = b_x[cross_point[0]:cross_point[1]]
            # a_t[t,:self.num_partitions] = a_x

            # # crossover y: execution_order
            # for j in range(cross_point[0], cross_point[1]):
            #     for k in range(self.num_partitions):
            #         if b_y[j] == a_y[k]:
            #             temp = a_y[j]
            #             a_y[j] = a_y[k]
            #             a_y[k] = temp
            #             break
            # a_t[t,self.num_partitions:] = a_y
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
            x = v_t[t,:self.num_partitions]
            y = v_t[t,self.num_partitions:]
            mutation_point = np.random.randint(low=0, high=self.num_partitions)

            # mutate x: deployed_server
            another_s_id = np.random.choice(self.server_lst)
            x[mutation_point] = another_s_id
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
    def deployed_server_reparation(self, x, y):
        server_lst = deepcopy(self.server_lst)
        cloud_lst = list(self.system_manager.cloud.keys())
        # request device constraint
        for c_id, s_id in enumerate(x):
            if s_id not in server_lst:
                x[c_id] = self.dataset.piece_device_map[c_id]
        self.system_manager.set_env(deployed_server=x, execution_order=y)
        while False in self.system_manager.constraint_chk():
            # 각 서버에 대해서,
            for s_id in range(self.num_servers-1):
                deployed_container_lst = list(np.where(x == s_id)[0])
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                self.system_manager.set_env(deployed_server=x, execution_order=y)
                while self.system_manager.constraint_chk(s_id=s_id) == False:
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    x[c_id] = self.server_lst[-1] # edge server로 보냄
                    self.system_manager.set_env(deployed_server=x, execution_order=y)

                # while self.system_manager.constraint_chk(s_id=s_id) == False:
                #     # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                #     c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                #     random.shuffle(server_lst)
                #     for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                #         if another_s_id == 0:
                #             another_s_id = self.dataset.piece_device_map[c_id]
                #         self.system_manager.set_env(deployed_server=x, execution_order=y)
                #         if s_id != another_s_id and self.system_manager.server[another_s_id].cur_energy > 0 and self.system_manager.constraint_chk(s_id=another_s_id):
                #             x[c_id] = another_s_id # 한번 넣어보고
                #             self.system_manager.set_env(deployed_server=x, execution_order=y)
                #             if self.system_manager.constraint_chk(s_id=another_s_id): # 자원 넘치는지 확인.
                #                 break
                #             else:
                #                 x[c_id] = s_id # 자원 넘치면 롤백

    # convert invalid action to valid action, multiprocessing function.
    def reparation_multiprocessing(self, inputs):
        x, y = inputs
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(x[t], y[t])
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.concatenate([x, y], axis=1)

    # convert invalid action to valid action.
    def reparation(self, positions):
        working_queue = [(position[:,:self.num_partitions], position[:,self.num_partitions:]) for position in positions]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.reparation_multiprocessing, working_queue))
        positions = np.array(temp)
        return positions
    
    def evaluation_multiprocessing(self, inputs):
        x, y = inputs
        self.system_manager.init_env()
        reward = []
        for t in range(self.num_timeslots):
            self.system_manager.set_env(deployed_server=x[t], execution_order=y[t])
            reward.append(self.system_manager.get_reward())
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return reward

    def evaluation(self, positions):
        working_queue = [(position[:,:self.num_partitions], position[:,self.num_partitions:]) for position in positions]
        with mp.Pool(processes=30) as pool:
            evaluation_lst = list(pool.map(self.evaluation_multiprocessing, working_queue))
        evaluation_lst = np.array(evaluation_lst)
        return evaluation_lst


class PSOGA:
    def __init__(self, dataset, num_particles, w_max=1.0, w_min=0.5, c1_s=0.5, c1_e=0.5, c2_s=0.5, c2_e=0.5):
        self.num_particles = num_particles
        self.dataset = dataset
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = dataset.num_partitions

        self.coarsened_graph = deepcopy(dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.w_max = w_max            # constant inertia max weight (how much to weigh the previous velocity)
        self.w_min = w_min            # constant inertia min weight (how much to weigh the previous velocity)
        self.c1_s = c1_s                  # cognative constant (start)
        self.c1_e = c1_e                  # cognative constant (end)
        self.c2_s = c2_s                  # social constant (start)
        self.c2_e = c2_e                  # social constant (end)

    def init(self):
        self.coarsened_graph = deepcopy(self.dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
    
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

    def update_w(self, x_t, g_t):
        ps = 0
        for i in range(self.num_particles):
            ps += np.sum(np.equal(x_t[i], g_t), axis=None)
        ps /= self.num_particles * self.num_timeslots * self.num_pieces
        w = self.w_max - (self.w_max - self.w_min) * np.exp(ps / (ps - 1.01))
        return w

    def refinement(self, x, refinement):
        if not refinement:
            return self.get_uncoarsened_x(x)
        start = time.time()
        self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
        print("refinement before:", max(self.system_manager.total_time_dp()))
        for i in range(3):
            new_coarsened_graph = []
            new_x = np.zeros(len(x)*2)
            for idx, cg in enumerate(self.coarsened_graph):
                num_pieces = len(self.graph[idx])
                for i in range(num_pieces):
                    indices = np.where(cg == i)[0]
                    cg[indices[int(len(indices)/2):]] += num_pieces # twice
                new_x[idx*num_pieces*2:(idx+1)*num_pieces*2] = np.concatenate([x[idx*num_pieces:(idx+1)*num_pieces], x[idx*num_pieces:(idx+1)*num_pieces]], axis=0)
                new_coarsened_graph.append(cg)
            x = new_x
            self.coarsened_graph = new_coarsened_graph
            self.graph = [np.unique(cg) for cg in self.coarsened_graph]
            self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
            self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])

            # new_coarsened_graph = []
            # x = self.get_uncoarsened_x(x)
            # for idx in range(self.num_services):
            #     cg = np.where(self.dataset.partition_service_map == idx)[0]
            #     new_coarsened_graph.append(cg)
            # self.coarsened_graph = new_coarsened_graph
            # self.graph = [np.unique(cg) for cg in self.coarsened_graph]
            # self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])

            # local search x: deployed_server
            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
            max_reward = self.system_manager.get_reward()
            print(-max_reward, time.time() - start)

            for j in np.arange(self.num_pieces): # for jth piece
                for s_id in self.server_lst:
                    if s_id == 0:
                        s_id = self.piece_device_map[j]
                    temp = x[j]
                    x[j] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                    cur_reward = self.system_manager.get_reward()
                    if cur_reward > max_reward and all(self.system_manager.constraint_chk()):
                        max_reward = cur_reward
                    else:
                        x[j] = temp

        print("took:", time.time() - start)
        return self.get_uncoarsened_x(x)

    def generate_random_solutions(self, greedy_solution=None, step=2):
        return np.random.choice(self.server_lst, size=(self.num_particles, self.num_timeslots, self.num_pieces))
        # return np.full(shape=(self.num_particles, self.num_timeslots, self.num_pieces), fill_value=self.server_lst[-1])
        # return np.array([[self.dataset.piece_device_map for _ in range(self.num_timeslots)] for _ in range(0, self.num_particles)])
        random_solutions = np.zeros((self.num_particles, self.num_timeslots, self.num_pieces))
        start = end = 0
        for i in self.server_lst:
            start = end
            end += step
            random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_pieces), fill_value=i)
        random_solutions[end:self.num_particles,:,:] = np.random.choice(self.server_lst, size=(self.num_particles-end, self.num_timeslots, self.num_pieces))
        return random_solutions
    
    def local_search_multiprocessing(self, action):
        np.random.seed(random.randint(0,2147483647))
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            # self.deployed_server_reparation(action[t])

            # local search x: deployed_server
            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]))
            max_reward = self.system_manager.get_reward()
            for j in np.random.choice(self.num_pieces, size=math.ceil(self.num_pieces/5), replace=False): # for jth layer # math.ceil(self.num_pieces/5)
                for s_id in self.server_lst:
                    if s_id == 0:
                        s_id = self.dataset.piece_device_map[j]
                    temp = action[t,j]
                    action[t,j] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(action[t]))
                    cur_reward = self.system_manager.get_reward()
                    if cur_reward > max_reward and all(self.system_manager.constraint_chk()):
                        max_reward = cur_reward
                    else:
                        action[t,j] = temp

            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(action[t]), timeslot=t)
        return action

    # we have to find local optimum from current chromosomes.
    def local_search(self, action, local_prob=0.5):
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

    def run_algo(self, loop, verbose=True, local_search=False, refinement=False, greedy_solution=None, early_exit_loop=5):
        self.init() # for refinement
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        x_t = self.generate_random_solutions(greedy_solution)
        x_t = self.reparation(x_t)
        if local_search:
            x_t = self.local_search(x_t, local_prob=0.5)
        p_t, g_t, p_t_eval_lst = self.selection(x_t)

        for i in range(loop):
            v_t = self.mutation(x_t, g_t)
            c_t = self.crossover(v_t, p_t, crossover_rate=(self.c1_e-self.c1_s) * (i / loop) + self.c1_s)
            x_t = self.crossover(c_t, g_t, crossover_rate=(self.c2_e-self.c2_s) * (i / loop) + self.c2_s, is_global=True)
            x_t = self.reparation(x_t)
            
            if local_search:
                x_t = self.local_search(x_t, local_prob=0.5)
            p_t, g_t, p_t_eval_lst = self.selection(x_t, p_t, g_t, p_t_eval_lst=p_t_eval_lst)
            eval_lst.append(np.max(np.sum(p_t_eval_lst, axis=1)))

            if max_reward < np.max(np.sum(p_t_eval_lst, axis=1)):
                max_reward = np.max(np.sum(p_t_eval_lst, axis=1))
                not_changed_loop = 0
            elif not_changed_loop > early_exit_loop:
                uncoarsened_x = np.array([self.refinement(g_t[t], refinement) for t in range(self.num_timeslots)])
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
                print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))

        uncoarsened_x = np.array([self.refinement(g_t[t], refinement) for t in range(self.num_timeslots)])
        self.system_manager.set_env(deployed_server=uncoarsened_x[0])
        t = self.system_manager.total_time_dp()
        e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
        print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
        print(g_t[0])
        print("t:", max(t), t)
        print("e:", sum(e), e)
        print("r:", r, "\033[0m")
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
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t = inputs
        for t in range(self.num_timeslots):
            cross_point = np.random.randint(low=0, high=self.num_pieces, size=math.ceil(self.num_pieces/5))

            # crossover x: deployed_server
            for cp in cross_point:
                a_t[t,cp] = b_t[t,cp]
        return a_t
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t = input
        for t in range(self.num_timeslots):
            a_x = a_t[t]
            b_x = b_t[t]
            cross_point = np.random.randint(low=0, high=a_x.size, size=2)

            # crossover x: deployed_server
            a_x[cross_point[0]:cross_point[1]] = b_x[cross_point[0]:cross_point[1]]
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

    # convert invalid action to valid action, deployed server action.
    def deployed_server_reparation(self, x):
        server_lst = deepcopy(self.server_lst)
        cloud_lst = list(self.system_manager.cloud.keys())
        # request device constraint
        for c_id, s_id in enumerate(x):
            if s_id not in server_lst:
                x[c_id] = self.dataset.piece_device_map[c_id]
        self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
        while False in self.system_manager.constraint_chk():
            # 각 서버에 대해서,
            for s_id in range(self.num_servers-1):
                deployed_container_lst = list(np.where(x == s_id)[0])
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                while self.system_manager.constraint_chk(s_id=s_id) == False:
                    # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        if another_s_id == 0:
                            another_s_id = self.dataset.piece_device_map[c_id]
                        self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                        if s_id != another_s_id and self.system_manager.server[another_s_id].cur_energy > 0 and self.system_manager.constraint_chk(s_id=another_s_id):
                            x[c_id] = another_s_id # 한번 넣어보고
                            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                            if self.system_manager.constraint_chk(s_id=another_s_id): # 자원 넘치는지 확인.
                                break
                            else:
                                x[c_id] = s_id # 자원 넘치면 롤백

    # convert invalid action to valid action, multiprocessing function.
    def reparation_multiprocessing(self, position):
        self.system_manager.init_env()
        for t in range(self.num_timeslots):
            self.deployed_server_reparation(position[t])
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(position[t]), timeslot=t)
        return position

    # convert invalid action to valid action.
    def reparation(self, positions):
        working_queue = [position for position in positions]
        with mp.Pool(processes=30) as pool:
            temp = list(pool.map(self.reparation_multiprocessing, working_queue))
        positions = np.array(temp)
        return positions
    
    def evaluation_multiprocessing(self, position):
        self.system_manager.init_env()
        reward = []
        for t in range(self.num_timeslots):
            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(position[t]))
            reward.append(self.system_manager.get_reward())
            self.system_manager.after_timeslot(deployed_server=self.get_uncoarsened_x(position[t]), timeslot=t)
        return reward

    def evaluation(self, positions):
        working_queue = [position for position in positions]
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
        self.num_services = dataset.num_services
        self.num_timeslots = dataset.num_timeslots
        self.num_partitions = dataset.num_partitions

        self.coarsened_graph = deepcopy(dataset.coarsened_graph)
        self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])

        self.server_lst = list(self.system_manager.local.keys()) + list(self.system_manager.edge.keys())

        self.mutation_ratio = mutation_ratio
        self.cross_over_ratio = cross_over_ratio

    def generate_random_solutions(self, greedy_solution=None, step=2):
        return np.random.choice(self.server_lst, size=(self.num_solutions, self.num_timeslots, self.num_pieces))
        # return np.full(shape=(self.num_solutions, self.num_timeslots, self.num_pieces), fill_value=self.server_lst[-1])
        # return np.array([[self.dataset.piece_device_map for _ in range(self.num_timeslots)] for _ in range(0, self.num_solutions)])
        random_solutions = np.zeros((self.num_solutions, self.num_timeslots, self.num_pieces))
        start = end = 0
        for i in self.server_lst:
            start = end
            end += step
            random_solutions[start:end,:,:] = np.full(shape=(end-start, self.num_timeslots, self.num_pieces), fill_value=i)
        random_solutions[end:self.num_solutions,:,:] = np.random.choice(self.server_lst, size=(self.num_solutions-end, self.num_timeslots, self.num_pieces))
        return random_solutions

    def run_algo(self, loop, verbose=True, local_search=False, refinement=False, greedy_solution=None, early_exit_loop=5):
        self.init() # for refinement
        start = time.time()
        max_reward = -np.inf
        not_changed_loop = 0
        eval_lst = []

        p_t = self.generate_random_solutions(greedy_solution)
        p_t = self.reparation(p_t)

        p_known = np.copy(p_t)
        if local_search:
            p_t = self.local_search(p_t, local_prob=0.5)

        for i in range(loop):
            q_t = self.selection(p_t, p_known)
            q_t = self.mutation(q_t, self.mutation_ratio)
            q_t = self.crossover(q_t, self.cross_over_ratio)
            q_t = self.reparation(q_t)

            if local_search:
                q_t = self.local_search(q_t, local_prob=0.5)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(p_t, q_t)
            eval_lst.append(v)

            if max_reward < v:
                max_reward = v
                not_changed_loop = 0
            elif not_changed_loop > early_exit_loop:
                uncoarsened_x = np.array([self.refinement(p_t[0,t], refinement) for t in range(self.num_timeslots)])
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
                print("mean t: {:.5f}".format(np.max(total_time, axis=None)), np.max(np.array(total_time), axis=0))
                print("mean e: {:.5f}".format(sum(total_energy) / self.num_timeslots))
                print("mean r: {:.5f}".format(sum(total_reward) / self.num_timeslots))
                print("avg took: {:.5f} sec".format((end - start) / (i + 1)))
                print("total took: {:.5f} sec".format(end - start))

        uncoarsened_x = np.array([self.refinement(p_t[0,t], refinement) for t in range(self.num_timeslots)])
        self.system_manager.set_env(deployed_server=uncoarsened_x[0])
        t = self.system_manager.total_time_dp()
        e = [s.energy_consumption() for s in list(self.system_manager.request.values()) + list(self.system_manager.local.values()) + list(self.system_manager.edge.values())]
        print("\033[31mEarly exit loop {}: {:.5f} sec".format(i + 1, time.time() - start))
        print(p_t[0,0])
        print("t:", max(t), t)
        print("e:", sum(e), e)
        print("r:", r, "\033[0m")
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
            cross_point = np.random.randint(low=0, high=self.num_pieces, size=math.ceil(self.num_pieces/5))

            # crossover x: deployed_server
            for cp in cross_point:
                temp = a_t[t,cp]
                a_t[t,cp] = b_t[t,cp]
                b_t[t,cp] = temp
        return np.concatenate([a_t.reshape(1,self.num_timeslots,-1), b_t.reshape(1,self.num_timeslots,-1)], axis=0)
        np.random.seed(random.randint(0,2147483647))
        a_t, b_t = inputs
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
            self.system_manager.after_timeslot(deployed_server=x[t], execution_order=y[t], timeslot=t)
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


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
        self.init()
        timer = time.time()
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

        # piece_rank = [max(rank[np.where(partition_piece_map==idx)]) for idx, x_i in enumerate(x)]
        # piece_order = np.array(sorted(zip(piece_rank, np.arange(self.num_pieces)), reverse=True), dtype=np.int32)[:,1]

        # for piece priority
        piece_rank = [max(rank[np.where(partition_piece_map==idx)]) for idx, x_i in enumerate(x)]
        piece_service_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])
        piece_order = []
        while sum(piece_rank) > 0:
            max_idx = np.argmax(piece_rank)
            temp_lst = np.where(piece_service_map==piece_service_map[max_idx])[0]
            temp_rank = rank[temp_lst]
            for i in reversed(np.argsort(temp_rank)):
                piece_order.append(temp_lst[i])
                piece_rank[temp_lst[i]] = 0

        for p_id in piece_order:
            minimum_latency = np.inf
            for s_id in self.server_lst:
                if s_id == 0:
                    s_id = self.piece_device_map[p_id]
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

        for i in range(4):
            # divide and conquer
            if i < 3:
                new_coarsened_graph = []
                new_x = np.zeros(len(x)*2)
                for idx, cg in enumerate(self.coarsened_graph):
                    num_pieces = len(self.graph[idx])
                    for piece in range(num_pieces):
                        indices = np.where(cg == piece)[0]
                        cg[indices[int(len(indices)/2):]] += num_pieces # twice
                    new_x[idx*num_pieces*2:(idx+1)*num_pieces*2] = np.concatenate([x[idx*num_pieces:(idx+1)*num_pieces], x[idx*num_pieces:(idx+1)*num_pieces]], axis=0)
                    new_coarsened_graph.append(cg)
                x = new_x
                self.coarsened_graph = new_coarsened_graph
                self.graph = [np.unique(cg) for cg in self.coarsened_graph]
                self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
                self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])

                idx_start = idx_end = start = end = 0
                for cg in self.coarsened_graph:
                    start = end
                    end += len(np.unique(cg))
                    idx_start = idx_end
                    idx_end += len(cg)
                    for idx, p in enumerate(cg):
                        partition_piece_map[idx_start + idx] = start + p

                # piece_rank = [max(rank[np.where(partition_piece_map==idx)]) for idx, _ in enumerate(x)]
                # piece_order = np.array(sorted(zip(piece_rank, np.arange(self.num_pieces)), reverse=True), dtype=np.int32)[:,1]

                # for piece priority
                piece_rank = [max(rank[np.where(partition_piece_map==idx)]) for idx, _ in enumerate(x)]
                piece_service_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])
                piece_order = []
                while sum(piece_rank) > 0:
                    max_idx = np.argmax(piece_rank)
                    temp_lst = np.where(piece_service_map==piece_service_map[max_idx])[0]
                    temp_rank = rank[temp_lst]
                    for i in reversed(np.argsort(temp_rank)):
                        piece_order.append(temp_lst[i])
                        piece_rank[temp_lst[i]] = 0
            else:
                x = self.get_uncoarsened_x(x)
                self.coarsened_graph = [np.arange(len(svc.partitions)) for svc in self.dataset.svc_set.services]
                self.graph = [np.unique(cg) for cg in self.coarsened_graph]
                self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
                self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])

                partition_piece_map = np.arange(self.num_partitions)
                piece_order = np.array(sorted(zip(rank, np.arange(self.num_pieces)), reverse=True), dtype=np.int32)[:,1]

            self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
            minimum_latency = max(self.system_manager.total_time_dp())
            print(minimum_latency, time.time() - timer)

            for j in piece_order: # for jth piece
                for s_id in set(self.server_lst) - set([x[j]]):
                    if s_id == 0:
                        s_id = self.piece_device_map[j]
                    temp = x[j]
                    x[j] = s_id
                    self.system_manager.set_env(deployed_server=self.get_uncoarsened_x(x))
                    latency = max(self.system_manager.total_time_dp())
                    if latency < minimum_latency and all(self.system_manager.constraint_chk()):
                        minimum_latency = latency
                    else:
                        x[j] = temp

        print("took:", time.time() - timer)
        return np.array([self.get_uncoarsened_x(x) for _ in range(self.num_timeslots)], dtype=np.int32), np.array([y for _ in range(self.num_timeslots)], dtype=np.int32)