import numpy as np
import math
import random
import dag_data_generator

import multiprocessing as mp
import time


class PSOGA:
    def __init__(self, dataset, num_particles, w_max=1.0, w_min=0.8, c1=0.8, c2=0.8):
        self.num_particles = num_particles
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_partitions = dataset.num_partitions

        self.w_max = w_max            # constant inertia max weight (how much to weigh the previous velocity)
        self.w_min = w_min            # constant inertia min weight (how much to weigh the previous velocity)
        self.c1 = c1                  # cognative constant
        self.c2 = c2                  # social constant

    def update_w(self, x_t, g_t):
        ps = 0
        for i in range(self.num_particles):
            ps += sum(np.equal(x_t[i], g_t))
        ps /=  self.num_partitions * self.num_particles
        w = self.w_max - (self.w_max - self.w_min) * np.exp(ps / (ps - 1.01))
        return w

    def generate_initial_particles(self):
        x_lst = np.random.randint(low=0, high=self.num_servers-1, size=(self.num_particles, self.num_partitions))
        y_lst = []
        for _ in range(self.num_particles):
            temp = list(range(self.num_partitions))
            random.shuffle(temp)
            y_lst.append(temp)
        y_lst = np.array(y_lst)
        position = np.concatenate([x_lst, y_lst], axis=-1)
        return position

    # convert invalid action to valid action.
    def reparation(self, position):
        x_lst = np.array(position[:,:self.num_partitions])
        y_lst = np.array(position[:,self.num_partitions:])
        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        cloud_lst = list(self.system_manager.cloud.keys())
        for i in range(x_lst.shape[0]):
            # 각 서버에 대해서,
            for s_id in range(self.num_servers):
                deployed_container_lst = np.where(x_lst[i] == s_id)[0].tolist()
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                done = False
                while self.system_manager.constraint_chk(deployed_server=x_lst[i], s_id=s_id) == False:
                    # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        if self.system_manager.constraint_chk(deployed_server=x_lst[i], s_id=another_s_id):
                            x_lst[i,c_id] = another_s_id # 한번 넣어보고
                            if self.system_manager.constraint_chk(deployed_server=x_lst[i], s_id=another_s_id): # 자원 넘치는지 확인.
                                done = True
                                break
                            else:
                                x_lst[i,c_id] = s_id # 자원 넘치면 롤백
                    if done:
                        break
        position = np.concatenate([x_lst, y_lst], axis=-1)
        return position

    def run_algo(self, loop, timeslot=0, verbose=True):
        x_t = self.generate_initial_particles()
        x_t = self.reparation(x_t)
        p_t, g_t, eval_lst = self.selection(x_t, timeslot=timeslot)

        for i in range(loop):
            v_t = self.mutation(x_t, g_t)
            c_t = self.cross_over(v_t, p_t, crossover_rate=self.c1)
            x_t = self.cross_over(c_t, g_t, crossover_rate=self.c2, is_global=True)
            x_t = self.reparation(x_t)
            p_t, g_t, eval_lst = self.selection(x_t, p_t, g_t, eval_lst, timeslot=timeslot)

            # test
            x = np.array(g_t[:self.num_partitions])
            y = np.array(g_t[self.num_partitions:])
            self.system_manager.set_env(deployed_server=x, execution_order=y)
            if verbose:
                print("---------- #{} loop: {:.5f} ----------".format(i, self.system_manager.get_reward()))
                print("x: ", x)
                print("y: ", y)
                print("constraint", [s.constraint_chk() for s in self.system_manager.server.values()])
                print("m", [(s.memory - sum(s.deployed_partition_mem.values())) / s.memory for s in self.system_manager.server.values()])
                print("e: ", [s.energy - s.energy_consumption() for s in self.system_manager.server.values()])
                print("t: ", self.system_manager.total_time())
        return x, y

    def selection(self, x_t, p_t=None, g_t=None, p_t_eval_lst=None, timeslot=0):
        if p_t is None and g_t is None:
            p_t_eval_lst = self.evaluation(x_t, timeslot)
            p_t = np.copy(x_t)
            g_t = np.copy(x_t[np.argmax(p_t_eval_lst),:])
        else:
            new_eval_lst = self.evaluation(x_t, timeslot)
            indices = np.where(new_eval_lst > p_t_eval_lst)
            p_t[indices,:] = x_t[indices,:]
            p_t_eval_lst[indices] = new_eval_lst[indices]
            g_t = np.copy(p_t[np.argmax(p_t_eval_lst),:])
        return p_t, g_t, p_t_eval_lst

    def cross_over(self, a_t, b_t, crossover_rate, is_global=False):
        new_a_t = np.copy(a_t)
        for i in range(self.num_particles):
            if np.random.random() < crossover_rate:
                a_x = new_a_t[i,:self.num_partitions]
                a_y = new_a_t[i,self.num_partitions:]
                if is_global:
                    b_x = b_t[:self.num_partitions]
                    b_y = b_t[self.num_partitions:]
                else:
                    b_x = b_t[i,:self.num_partitions]
                    b_y = b_t[i,self.num_partitions:]

                cross_point = np.random.randint(low=0, high=a_x.size, size=2)

                # crossover x: deployed_server
                a_x[cross_point[0]:cross_point[1]] = b_x[cross_point[0]:cross_point[1]]
                new_a_t[i,:self.num_partitions] = a_x

                # crossover y: execution_order
                for j in range(cross_point[0], cross_point[1]):
                    for k in range(self.num_partitions):
                        if b_y[j] == a_y[k]:
                            temp = a_y[j]
                            a_y[j] = a_y[k]
                            a_y[k] = temp
                            break
                new_a_t[i,self.num_partitions:] = a_y
        return new_a_t

    def mutation(self, x_t, g_t, mutation_ratio=None):
        v_t = np.copy(x_t)
        if mutation_ratio == None:
            w = self.update_w(v_t[:,:self.num_partitions], g_t[:self.num_partitions])
        else:
            w = mutation_ratio
        for i in range(self.num_particles):
            if np.random.random() < w:
                x = v_t[i,:self.num_partitions]
                y = v_t[i,self.num_partitions:]

                mutation_point = np.random.randint(low=0, high=self.num_partitions)

                # mutate x: deployed_server
                x[mutation_point] = np.random.randint(low=0, high=self.num_servers-1)
                v_t[i,:self.num_partitions] = x

                # mutate y: execution_order
                rand = np.random.randint(low=0, high=self.num_partitions)
                for k in range(self.num_partitions):
                    if rand == y[k]:
                        temp = y[mutation_point]
                        y[mutation_point] = y[k]
                        y[k] = temp
                        break
                v_t[i,self.num_partitions:] = y
        return v_t
    
    def eval_multiprocessing(self, input):
        position, timeslot = input
        x = np.array(position[:self.num_partitions])
        y = np.array(position[self.num_partitions:])
        self.system_manager.set_env(deployed_server=x, execution_order=y)
        return self.system_manager.get_reward()

    def evaluation(self, positions, timeslot=0):
        working_queue = [(position, timeslot) for position in positions]
        with mp.Pool(processes=10) as pool:
            evaluation_lst = list(pool.map(self.eval_multiprocessing, working_queue))
        evaluation_lst = np.array(evaluation_lst)
        return evaluation_lst


class Genetic:
    def __init__(self, dataset, num_solutions):
        self.num_solutions = num_solutions
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_partitions = dataset.num_partitions
        self.num_genes = self.num_partitions * 2

    def generate_random_solutions(self):
        x_lst = np.random.randint(low=0, high=self.num_servers-1, size=(self.num_solutions, self.num_partitions))
        y_lst = []
        for _ in range(self.num_solutions):
            temp = list(range(self.num_partitions))
            random.shuffle(temp)
            y_lst.append(temp)
        y_lst = np.array(y_lst)
        action = np.concatenate([x_lst, y_lst], axis=-1)
        return action

    # convert invalid action to valid action.
    def reparation(self, action):
        x_lst = np.array(action[:,:self.num_partitions])
        y_lst = np.array(action[:,self.num_partitions:])

        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        cloud_lst = list(self.system_manager.cloud.keys())
        for i in range(x_lst.shape[0]):
            # 각 서버에 대해서,
            for s_id in range(self.num_servers):
                deployed_container_lst = np.where(x_lst[i] == s_id)[0].tolist()
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                done = False
                while self.system_manager.constraint_chk(deployed_server=x_lst[i], s_id=s_id) == False:
                    # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        if self.system_manager.constraint_chk(deployed_server=x_lst[i], s_id=another_s_id):
                            x_lst[i,c_id] = another_s_id # 한번 넣어보고
                            if self.system_manager.constraint_chk(deployed_server=x_lst[i], s_id=another_s_id): # 자원 넘치는지 확인.
                                done = True
                                break
                            else:
                                x_lst[i,c_id] = s_id # 자원 넘치면 롤백
                    if done:
                        break

        action = np.concatenate([x_lst, y_lst], axis=-1)
        return action
    
    def local_search_multiprocessing(self, input):
        x, y, scheduling_lst = input
        for j in scheduling_lst: # for jth layer
            # local search x: deployed_server
            c = self.system_manager.service_set.partitions[j]
            self.system_manager.set_env(deployed_server=x, execution_order=y)
            min_time = self.system_manager.total_time()
            s_set = set()
            if len(c.predecessors):
                for p in c.predecessors:
                    s_set.add(x[p.id])
            else:
                for s_id in range(self.num_servers-1):
                    s_set.add(s_id)
                
            for s_id in s_set:
                if x[j] == s_id:
                    continue
                temp = x[j]
                x[j] = s_id
                self.system_manager.set_env(deployed_server=x, execution_order=y)
                time = self.system_manager.total_time()
                if self.system_manager.constraint_chk(deployed_server=x, s_id=s_id) and time < min_time:
                    min_time = time
                else:
                    x[j] = temp
        return x

    # we have to find local optimum from current chromosomes.
    def local_search(self, action, timeslot=0, local_prob=0.2):
        x_lst = np.array(action[:,:self.num_partitions])
        y_lst = np.array(action[:,self.num_partitions:])

        local_idx = np.random.rand(action.shape[0])
        local_idx = local_idx < local_prob
        local_idx = np.where(local_idx)[0]
        np.random.shuffle(local_idx)

        scheduling_lst = np.array(sorted(zip(self.system_manager.ranku, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1]

        working_queue = [(x_lst[i], y_lst[i], scheduling_lst) for i in local_idx]
        with mp.Pool(processes=10) as pool:
            temp = list(pool.map(self.local_search_multiprocessing, working_queue))
        x_lst[local_idx] = np.array(temp)

        action = np.concatenate([x_lst, y_lst], axis=-1)
        return action

    def run_algo(self, loop, mutation_ratio=0.5, cross_over_ratio=0.5, timeslot=0, verbose=True):
        ev_lst = np.zeros(loop, dtype=np.float_)
        p_t = self.generate_random_solutions()
        p_t = self.reparation(p_t)

        p_known = np.copy(p_t)
        p_t = self.local_search(p_t, timeslot=0, local_prob=0.5)

        for i in range(loop):
            start = time.time()
            q_t = self.selection(p_t, p_known)
            print("selection", time.time() - start)

            q_t = self.cross_over(q_t, cross_over_ratio)
            print("cross_over", time.time() - start)
            q_t = self.mutation(q_t, mutation_ratio)
            print("mutation", time.time() - start)
            q_t = self.reparation(q_t)
            print("reparation", time.time() - start)

            q_t = self.local_search(q_t, timeslot=0, local_prob=0.5)
            print("local_search", time.time() - start)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(timeslot, p_t, q_t)
            ev_lst[i] = v
            print("fitness_selection", time.time() - start)

            # test
            x = np.array(p_t[0,:self.num_partitions])
            y = np.array(p_t[0,self.num_partitions:])
            self.system_manager.set_env(deployed_server=x, execution_order=y)
            if verbose:
                print("---------- #{} loop: {:.5f} ----------".format(i, self.system_manager.get_reward()))
                print("x: ", x)
                print("y: ", y)
                print("constraint", [s.constraint_chk() for s in self.system_manager.server.values()])
                print("m", [(s.memory - sum(s.deployed_partition_mem.values())) / s.memory for s in self.system_manager.server.values()])
                print("e: ", [s.energy - s.energy_consumption() for s in self.system_manager.server.values()])
                print("t: ", self.system_manager.total_time())

        return x, y, ev_lst

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

    def fitness_selection(self, timeslot, *args):
        union, max_len = self.union(*args)

        ev_lst = self.evaluation(union, timeslot=timeslot)
        ev_lst = list(map(lambda x: (x[0], x[1]), enumerate(ev_lst)))
        ev_lst.sort(key=lambda x: x[1], reverse=True)
        sorted_idx = list(map(lambda x: x[0], ev_lst))
        union = union[sorted_idx[:max_len], :]
        return union, ev_lst[0][1]

    def cross_over(self, action, crossover_rate):
        x_lst = np.array(action[:,:self.num_partitions])
        y_lst = np.array(action[:,self.num_partitions:])

        crossover_idx = np.random.rand(action.shape[0])
        crossover_idx = crossover_idx < crossover_rate
        crossover_idx = np.where(crossover_idx)[0]
        np.random.shuffle(crossover_idx)

        for i in range(math.floor(crossover_idx.size / 2)):
            a_idx = crossover_idx[i * 2]
            b_idx = crossover_idx[i * 2 + 1]

            a_x = x_lst[a_idx, :]
            a_y = y_lst[a_idx, :]
            b_x = x_lst[b_idx, :]
            b_y = y_lst[b_idx, :]

            cross_point = np.random.randint(1, a_x.size - 1)

            # crossover x: deployed_server
            temp = a_x[:cross_point]
            a_x[:cross_point] = b_x[:cross_point]
            b_x[:cross_point] = temp
            x_lst[a_idx, :] = a_x
            x_lst[b_idx, :] = b_x

            # crossover y: execution_order
            temp_a_y = np.copy(a_y)
            temp_b_y = np.copy(b_y)
            a_y[:cross_point] = temp_b_y[:cross_point]
            for j in range(cross_point, a_y.size):
                for k in temp_a_y:
                    if k not in a_y[:j]:
                        a_y[j] = k
                        break
            b_y[:cross_point] = temp_a_y[:cross_point]
            for j in range(cross_point, b_y.size):
                for k in temp_b_y:
                    if k not in b_y[:j]:
                        b_y[j] = k
                        break
            y_lst[a_idx, :] = a_y
            y_lst[b_idx, :] = b_y

        action = np.concatenate([x_lst, y_lst], axis=-1)
        return action

    def mutation(self, action, mutation_ratio):
        x_lst = np.array(action[:,:self.num_partitions])
        y_lst = np.array(action[:,self.num_partitions:])

        mutation_idx = np.random.rand(action.shape[0])
        mutation_idx = mutation_idx < mutation_ratio
        mutation_idx = np.where(mutation_idx)[0]
        np.random.shuffle(mutation_idx)

        for i in mutation_idx:
            mutation_point = np.random.randint(low=0, high=self.num_partitions)
            # mutate x: deployed_server
            x_lst[i, mutation_point] = np.random.randint(low=0, high=self.num_servers-1)

            # mutate y: execution_order
            rand = np.random.randint(low=0, high=self.num_partitions)
            for j in range(self.num_partitions):
                if y_lst[i, j] == rand:
                    temp = y_lst[i, mutation_point]
                    y_lst[i, mutation_point] = y_lst[i, j]
                    y_lst[i, j] = temp
                    break

        action = np.concatenate([x_lst, y_lst], axis=-1)
        return action
    
    def evaluation_multiprocessing(self, input):
        action, timeslot = input
        x = np.array(action[:self.num_partitions])
        y = np.array(action[self.num_partitions:])
        self.system_manager.set_env(deployed_server=x, execution_order=y)
        return self.system_manager.get_reward()

    def evaluation(self, actions, timeslot=0, test_out=False):
        working_queue = [(action, timeslot) for action in actions]
        with mp.Pool(processes=10) as pool:
            evaluation_lst = list(pool.map(self.evaluation_multiprocessing, working_queue))
        evaluation_lst = np.array(evaluation_lst)
        return evaluation_lst


class HEFT:
    def __init__(self, dataset):
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_partitions = dataset.num_partitions

    def run_algo(self, timeslot=0):
        x = np.full(shape=self.num_partitions, fill_value=self.system_manager.cloud_id, dtype=np.int32)
        y = np.arange(stop=self.num_partitions, dtype=np.int32)
        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        scheduling_lst = np.array(sorted(zip(self.system_manager.ranku, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1]

        y = scheduling_lst
        for i, top_rank in enumerate(scheduling_lst):
            # initialize the earliest finish time of the task
            earliest_finish_time = np.inf
            # for all available server, find earliest finish time server
            for s_id in server_lst:
                temp_x = x[top_rank]
                x[top_rank] = s_id
                if self.system_manager.constraint_chk(x, s_id):
                    self.system_manager.set_env(deployed_server=x, execution_order=y)
                    finish_time = self.system_manager.get_completion_time(top_rank)
                    if finish_time < earliest_finish_time:
                        earliest_finish_time = finish_time
                    else:
                        x[top_rank] = temp_x
                else:
                    x[top_rank] = temp_x
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)


if __name__=="__main__":
    dataset = dag_data_generator.DAGDataSet(max_timeslot=24)

    greedy = HEFT(dataset=dataset)
    genetic = Genetic(dataset=dataset, num_solutions=200)
    psoga = PSOGA(dataset=dataset, num_particles=200, w_max=1.0, w_min=0.8, c1=0.8, c2=0.8)

    # start = time.time()
    # x, y = psoga.run_algo(loop=200, verbose=False)
    # dataset.system_manager.set_env(deployed_server=x, execution_order=y)
    # print("---------- PSO-GA Algorithm ----------")
    # print("x: ", x)
    # print("y: ", y)
    # print("constraint", [s.constraint_chk() for s in dataset.system_manager.server.values()])
    # print("m", [(s.memory - sum(s.deployed_partition_mem.values())) / s.memory for s in dataset.system_manager.server.values()])
    # print("e: ", [s.energy - s.energy_consumption() for s in dataset.system_manager.server.values()])
    # print("t: ", dataset.system_manager.total_time())
    # print("reward: {:.3f}".format(dataset.system_manager.get_reward()))
    # print("took: {:.3f} sec".format(time.time() - start))
    # print("---------- PSO-GA Algorithm ----------\n")

    start = time.time()
    x, y, v = genetic.run_algo(loop=100, mutation_ratio=0.2, cross_over_ratio=0.2, verbose=True)
    dataset.system_manager.set_env(deployed_server=x, execution_order=y)
    print("---------- Genetic Algorithm ----------")
    print("x: ", x)
    print("y: ", y)
    print("constraint", [s.constraint_chk() for s in dataset.system_manager.server.values()])
    print("m", [(s.memory - sum(s.deployed_partition_mem.values())) / s.memory for s in dataset.system_manager.server.values()])
    print("e: ", [s.energy - s.energy_consumption() for s in dataset.system_manager.server.values()])
    print("t: ", dataset.system_manager.total_time())
    print("reward: {:.3f}".format(dataset.system_manager.get_reward()))
    print("took: {:.3f} sec".format(time.time() - start))
    print("---------- Genetic Algorithm ----------\n")

    start = time.time()
    x, y = greedy.run_algo()
    dataset.system_manager.set_env(deployed_server=x, execution_order=y)
    print("---------- Greedy Algorithm ----------")
    print("x: ", x)
    print("y: ", y)
    print("constraint", [s.constraint_chk() for s in dataset.system_manager.server.values()])
    print("m", [(s.memory - sum(s.deployed_partition_mem.values())) / s.memory for s in dataset.system_manager.server.values()])
    print("e: ", [s.energy - s.energy_consumption() for s in dataset.system_manager.server.values()])
    print("t: ", dataset.system_manager.total_time())
    print("reward: {:.5f}".format(dataset.system_manager.get_reward()))
    print("took: {:.5f} sec".format(time.time() - start))
    print("---------- Greedy Algorithm ----------\n")