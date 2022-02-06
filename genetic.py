import numpy as np
import math
import random
import dag_data_generator
from collections import Counter

import time


class Genetic:
    def __init__(self, dataset, num_solutions):
        self.num_solutions = num_solutions
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_partitions = dataset.num_partitions

        self.x_bits = int(math.log(dataset.num_servers, 2)) + 1
        self.y_bits = int(math.log(dataset.num_partitions, 2)) + 1
        self.num_genes = self.num_partitions * (self.x_bits + self.y_bits)

    def generate_random_solutions(self, chromosome_size, num_solutions):
        return np.random.choice([True, False], size=(num_solutions, chromosome_size))

    # convert invalid action to valid action.
    def reparation(self, encoded_action):
        decoded_action = np.array([self.decoding(encoded_action[i]) for i in range(self.num_solutions)])
        x_lst = np.array(decoded_action[:,0,:])
        y_lst = np.array(decoded_action[:,1,:])

        predecessor_lst = [[p.id for p in self.system_manager.service_set.partitions[i].predecessors] for i in range(self.num_partitions)]

        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        cloud_lst = list(self.system_manager.cloud.keys())
        for i in range(x_lst.shape[0]):
            # 만약 invalid한 server가 선택된 경우, random 서버로 보냄.
            for j in range(x_lst[i].shape[0]):
                if x_lst[i,j] not in server_lst:
                    x_lst[i,j] = random.choice(server_lst)

            # 각 서버에 대해서,
            for s_id in range(self.num_servers):
                deployed_container_lst = np.where(x_lst[i] == s_id)[0].tolist()
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                while self.system_manager.constraint_chk(x_lst[i], s_id) == False:
                    # 해당 서버에 deployed되어있는 partition 중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # random partition 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        x_lst[i,c_id] = another_s_id # 한번 넣어보고
                        if self.system_manager.constraint_chk(x_lst[i], another_s_id): # 자원 넘치는지 확인.
                            break
                        else:
                            x_lst[i,c_id] = s_id # 자원 넘치면 롤백

            server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
            scheduling_lst = sorted(zip(self.system_manager.ranku, np.arange(self.num_partitions)))
            # ranku 높은 순서로 y를 수정함.
            for _ in range(self.num_partitions):
                not_changed = True
                for _, j in scheduling_lst:
                    # predecessor의 순서가 더 늦으면 자신의 순서를 predecessor의 순서와 swap
                    random.shuffle(predecessor_lst[j])
                    for p_id in predecessor_lst[j]:
                        if y_lst[i,j] < y_lst[i,p_id]:
                            swap = y_lst[i,j]
                            y_lst[i,j] = y_lst[i,p_id]
                            y_lst[i,p_id] = swap
                            not_changed = False
                if not_changed:
                    break

        return np.array([self.encoding(x_lst[i], y_lst[i]) for i in range(self.num_solutions)])

    # we have to find local optimum from current chromosomes.
    def local_search(self, encoded_action, local_prob=0.5):
        decoded_action = np.array([self.decoding(encoded_action[i]) for i in range(self.num_solutions)])
        x_lst = np.array(decoded_action[:,0,:])
        y_lst = np.array(decoded_action[:,1,:])

        
        return np.array([self.encoding(x_lst[i], y_lst[i]) for i in range(self.num_solutions)])

    # gene encoding, y.shape = (num_partitions, ) / result.shape = (num_genes, )
    def encoding(self, x, y):
        action = np.zeros(self.num_genes, dtype=np.int32)
        for i in range(self.num_partitions):
            start = i * (self.x_bits + self.y_bits)
            end = start + self.x_bits
            action[start:end] = np.array([bool(x[i] & (1<<(n-1))) for n in range(self.x_bits, 0, -1)]) # integer to boolean array
            start = end
            end = start + self.y_bits
            action[start:end] = np.array([bool(y[i] & (1<<(n-1))) for n in range(self.y_bits, 0, -1)]) # integer to boolean array
        return action

    # gene decoding, y.shape = (num_genes, ) / result.shape = (num_partitions, )
    def decoding(self, encoded_action):
        x = np.zeros(self.num_partitions, dtype=np.int32)
        y = np.zeros(self.num_partitions, dtype=np.int32)
        for i in range(self.num_partitions):
            start = i * (self.x_bits + self.y_bits)
            x[i] = np.sum([encoded_action[start+idx] * (2**(n-1)) for idx, n in enumerate(range(self.x_bits, 0, -1))], 0) # boolean array to integer
            start += self.x_bits
            y[i] = np.sum([encoded_action[start+idx] * (2**(n-1)) for idx, n in enumerate(range(self.y_bits, 0, -1))], 0) # boolean array to integer
        return x, y

    def run_algo(self, loop, mutation_ratio=0.05, cross_over_ratio=0.05, timeslot=0):
        ev_lst = np.zeros(loop, dtype=np.float_)
        p_t = self.generate_random_solutions(self.num_genes, self.num_solutions)
        p_t = self.reparation(p_t)

        p_known = np.copy(p_t)
        p_t = self.local_search(p_t)

        for i in range(loop):
            q_t = self.selection(p_t, p_known)

            self.cross_over(q_t, cross_over_ratio)
            self.mutation(q_t, mutation_ratio)
            q_t = self.reparation(q_t)

            q_t = self.local_search(q_t)
            p_known = np.copy(q_t)
            p_t, v = self.fitness_selection(timeslot, p_t, q_t)
            ev_lst[i] = v

            # test
            x, y = self.decoding(p_t[0])
            self.system_manager.set_xy_mat(x, y)
            print("#{} loop: {}".format(i, self.system_manager.get_reward(timeslot=timeslot)))
            print(x, y)
            print([s.constraint_chk() for s in self.system_manager.server.values()])
            print("self.system_manager.total_time()", self.system_manager.total_time())

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

    @staticmethod
    def cross_over(mat, crossover_rate):
        if mat.shape[1] == 1:
            pass
        else:
            crossover_idx = np.random.rand(mat.shape[0])
            crossover_idx = crossover_idx < crossover_rate
            crossover_idx = np.where(crossover_idx)[0]
            np.random.shuffle(crossover_idx)

            for i in range(math.floor(crossover_idx.size / 2)):
                a_idx = i * 2
                b_idx = a_idx + 1

                a = mat[a_idx, :]
                b = mat[b_idx, :]
                k_shape = a.shape

                a = a.ravel()  # flatten
                b = b.ravel()

                cross_point = np.random.randint(1, a.size - 1) if a.size > 2 else 1
                temp = a[:cross_point]
                a[:cross_point] = b[:cross_point]
                b[:cross_point] = temp
                a = a.reshape(k_shape)
                b = b.reshape(k_shape)
                mat[a_idx, :] = a
                mat[b_idx, :] = b

    def mutation(self, mat, mutation_ratio):
        mut_idx = np.random.rand(*mat.shape)
        mut_idx = mut_idx < mutation_ratio
        mut_idx = np.where(mut_idx)
        mat[mut_idx] = ~mat[mut_idx].astype(bool)

    def evaluation(self, encoded_action, timeslot=0, test_out=False):
        decoded_action = np.array([self.decoding(encoded_action[i]) for i in range(self.num_solutions)])
        x_lst = np.array(decoded_action[:,0,:])
        y_lst = np.array(decoded_action[:,1,:])
        evaluation_lst = list()
        for i in range(self.num_solutions):
            self.system_manager.set_xy_mat(x_lst[i], y_lst[i])
            evaluation_lst.append(self.system_manager.get_reward(timeslot=timeslot))
        return evaluation_lst


class TraditionalGenetic(Genetic):
    def run_algo(self, loop, mutation_ratio=0.05, cross_over_ratio=0.05, timeslot=0):
        x_mat = self.system_manager.get_x()
        p_t = self.generate_random_solutions(self.num_genes, self.num_solutions)
        p_t = self.reparation(p_t)

        for i in range(loop):
            p_t = self.selection(p_t)

            # test
            self.system_manager.set_y_mat(self.decoding(p_t[0, :]))
            print("#{} loop: {}".format(i, self.system_manager.get_reward(timeslot=timeslot)))
            print(self.decoding(p_t[0, :]))
            print([s.constraint_chk() for s in self.system_manager.server.values()])
            print("self.system_manager.total_time()", self.system_manager.total_time())

            p_t = self.cross_over(p_t)
            self.mutation(p_t, mutation_ratio)

            p_t = self.reparation(p_t)

        p_t = self.selection(p_t)
        return self.decoding(p_t[0, :])

    def selection(self, solutions, rank_ratio=0.1):
        # use ranking
        # select high 10%
        max_len = int(self.num_solutions * rank_ratio)
        ev_lst = self.evaluation(solutions)
        ev_lst = list(map(lambda x: (x[0], x[1]), enumerate(ev_lst)))
        ev_lst.sort(key=lambda x: x[1], reverse=True)
        sorted_idx = list(map(lambda x: x[0], ev_lst))
        return solutions[sorted_idx[:max_len], :]

    def cross_over(self, parents):
        parents_idx = np.random.randint(0, parents.shape[0], size=(self.num_solutions, 2))
        select_parents = np.random.randint(0, 1, size=(self.num_solutions, self.num_genes))
        father = parents[parents_idx[:, 0], :]
        mother = parents[parents_idx[:, 1], :]
        return father * select_parents + mother * (1 - select_parents)


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
        scheduling_lst = np.array(sorted(zip(self.data_set.system_manager.ranku, np.arange(self.data_set.num_partitions)), reverse=True), dtype=np.int32)[:,1]

        for i, top_rank in enumerate(scheduling_lst):
            # initialize the earliest finish time of the task
            earliest_finish_time = np.inf
            y[top_rank] = i
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
    memetic = Genetic(dataset=dataset, num_solutions=100)
    genetic = TraditionalGenetic(dataset=dataset, num_solutions=100)

    start = time.time()
    x, y = greedy.run_algo()
    dataset.system_manager.set_xy_mat(x, y)
    print("---------- Greedy Algorithm ----------")
    print("x: ", dataset.system_manager._x)
    print("y: ", y)
    print([s.constraint_chk() for s in dataset.system_manager.server.values()])
    print("t: ", dataset.system_manager.total_time())
    print("reward: {:.3f}".format(dataset.system_manager.get_reward(timeslot=0)))
    print("took: {:.3f} sec".format(time.time() - start))
    print("---------- Greedy Algorithm ----------\n")
    input()

    # start = time.time()
    # y, v = genetic.run_algo(loop=1000, mutation_ratio=0.1, cross_over_ratio=0.1)
    # dataset.system_manager.set_y_mat(y)
    # print("---------- Genetic Algorithm ----------")
    # print("x: ", dataset.system_manager._x)
    # print("y: ", y)
    # print([s.constraint_chk() for s in dataset.system_manager.server.values()])
    # print("t: ", dataset.system_manager.total_time())
    # print("reward: {:.3f}".format(dataset.system_manager.get_reward(timeslot=0)))
    # print("took: {:.3f} sec".format(time.time() - start))
    # print("---------- Genetic Algorithm ----------\n")

    start = time.time()
    x, y, v = memetic.run_algo(loop=200, mutation_ratio=0.1, cross_over_ratio=0.1)
    dataset.system_manager.set_xy_mat(x, y)
    print("---------- Memetic Algorithm ----------")
    print("x: ", dataset.system_manager._x)
    print("y: ", y)
    print([s.constraint_chk() for s in dataset.system_manager.server.values()])
    print("t: ", dataset.system_manager.total_time())
    print("reward: {:.3f}".format(dataset.system_manager.get_reward(timeslot=0)))
    print("took: {:.3f} sec".format(time.time() - start))
    print("---------- Memetic Algorithm ----------\n")