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
        self.num_containers = dataset.num_containers

        self.encoding_bits = int(math.log(dataset.num_servers, 2)) + 1
        self.num_genes = self.num_containers * self.encoding_bits

    def generate_random_solutions(self, chromosome_size, num_solutions):
        return np.random.choice([True, False], size=(num_solutions, chromosome_size))

    # convert invalid action to valid action.
    def reparation(self, encoded_y_mat):
        y_mat = np.array([self.decoding(encoded_y_mat[i]) for i in range(self.num_solutions)])

        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        cloud_lst = list(self.system_manager.cloud.keys())
        for i in range(y_mat.shape[0]):
            # 만약 invalid한 server가 선택된 경우, random 서버로 보냄.
            for j in range(y_mat[i].shape[0]):
                if y_mat[i,j] not in server_lst:
                    y_mat[i,j] = random.choice(server_lst)

            # 각 서버에 대해서,
            for s_id in range(self.num_servers):
                deployed_container_lst = np.where(y_mat[i] == s_id)[0].tolist()
                random.shuffle(deployed_container_lst)
                # 서버가 넘치는 경우,
                while self.system_manager.constraint_chk(y_mat[i], s_id) == False:
                    # 해당 서버에 deployed되어있는 container중 하나를 자원이 충분한 랜덤 서버로 보냄.
                    c_id = deployed_container_lst.pop() # 아무 컨테이너 하나를 골라서
                    random.shuffle(server_lst)
                    for another_s_id in server_lst + cloud_lst: # 아무 서버에다가 (클라우드는 예외처리용임. 알고리즘에서는 넘치는걸 가정하지 않음.)
                        y_mat[i,c_id] = another_s_id # 한번 넣어보고
                        if self.system_manager.constraint_chk(y_mat[i], another_s_id): # 자원 넘치는지 확인.
                            break
                        else:
                            y_mat[i,c_id] = s_id # 자원 넘치면 롤백
        return np.array([self.encoding(y_mat[i]) for i in range(y_mat.shape[0])])

    # we have to find local optimum from current chromosomes.
    def local_search(self, encoded_y_mat, local_prob=0.5):
        y_mat = np.array([self.decoding(encoded_y_mat[i]) for i in range(self.num_solutions)])
        for i in range(y_mat.shape[0]):
            # 모든 container에 대해 predecessor와 successor가 위치한 server list를 얻고, 넣을 수 있는 곳에 집어넣음.
            self.system_manager.set_y_mat(y_mat[i])
            for c in self.system_manager.service_set.container_set:
                s_id = y_mat[i,c.id]
                new_s_id_lst = [y_mat[i,pred.id] for pred in c.predecessors] + [y_mat[i,succ.id] for succ in c.successors]
                for new_s_id in new_s_id_lst:
                    y_mat[i,c.id] = new_s_id # 한번 넣어보고
                    if self.system_manager.constraint_chk(y_mat[i], new_s_id) and new_s_id != self.system_manager.cloud_id: # 자원 넘치는지 확인.
                        break
                    else:
                        y_mat[i,c.id] = s_id # 자원 넘치면 롤백
        return np.array([self.encoding(y_mat[i]) for i in range(y_mat.shape[0])])

    # gene encoding, y.shape = (num_containers, ) / result.shape = (num_genes, )
    def encoding(self, y):
        result = np.zeros(self.num_genes, dtype=np.int32)
        for i in range(self.num_containers):
            start = i * self.encoding_bits
            end = start + self.encoding_bits
            result[start:end] = np.array([bool(y[i] & (1<<(n-1))) for n in range(self.encoding_bits, 0, -1)]) # integer to boolean array
        return result

    # gene decoding, y.shape = (num_genes, ) / result.shape = (num_containers, )
    def decoding(self, y):
        result = np.zeros(self.num_containers, dtype=np.int32)
        for i in range(self.num_containers):
            start = i * self.encoding_bits
            result[i] = np.sum([y[start+idx] * (2**(n-1)) for idx, n in enumerate(range(self.encoding_bits, 0, -1))], 0) # boolean array to integer
        return result

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
            self.system_manager.set_y_mat(self.decoding(p_t[0, :]))
            print("#{} loop: {}".format(i, self.system_manager.get_reward(timeslot=timeslot)))
            print(self.decoding(p_t[0, :]))
            print([s.constraint_chk() for s in self.system_manager.server.values()])
            print("self.system_manager.total_time()", self.system_manager.total_time())

        return self.decoding(p_t[0, :]), ev_lst

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

    def evaluation(self, y_lst, timeslot=0, test_out=False):
        evaluation_lst = list()
        for k_idx in range(y_lst.shape[0]):
            y = y_lst[k_idx, :]
            self.system_manager.set_y_mat(self.decoding(y))
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


class FirstFitDecreasing:
    def __init__(self, dataset):
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_containers = dataset.num_containers

    def run_algo(self, timeslot=0):
        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        server_mem_lst = [self.system_manager.server_mem[s_id] for s_id in server_lst]
        server_lst = [x for _, x in sorted(zip(server_mem_lst, server_lst), reverse=True)]
        y = np.full(shape=self.num_containers, fill_value=self.system_manager.cloud_id)
        container_id = 0
        for server_id in server_lst:
            while container_id < self.num_containers:
                y[container_id] = server_id
                container_id += 1
                if not self.system_manager.constraint_chk(y, server_id):
                    container_id -= 1
                    y[container_id] = self.system_manager.cloud_id
                    break
        return np.array(y, dtype=np.int32)


class FirstFitService:
    def __init__(self, dataset):
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_containers = dataset.num_containers

    def run_algo(self, timeslot=0):
        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        server_mem_lst = [self.system_manager.server_mem[s_id] for s_id in server_lst]
        server_lst = [x for _, x in sorted(zip(server_mem_lst, server_lst), reverse=True)]
        y = np.full(shape=self.num_containers, fill_value=self.system_manager.cloud_id)
        container_id = 0
        for service in self.system_manager.service_set.svc_set:
            for server_id in server_lst:
                c_id_lst = [c.id for c in service.partitions]
                y[c_id_lst] = server_id
                if self.system_manager.constraint_chk(y, server_id):
                    break
                else:
                    y[c_id_lst] = self.system_manager.cloud_id
        return np.array(y, dtype=np.int32)


if __name__=="__main__":
    dataset = dag_data_generator.DAGDataSet(max_timeslot=24)

    greedy = FirstFitDecreasing(dataset=dataset)
    heuristic = FirstFitService(dataset=dataset)
    memetic = Genetic(dataset=dataset, num_solutions=100)
    genetic = TraditionalGenetic(dataset=dataset, num_solutions=100)

    start = time.time()
    y = greedy.run_algo()
    dataset.system_manager.set_y_mat(y)
    print("---------- Greedy Algorithm ----------")
    print("x: ", dataset.system_manager._x)
    print("y: ", y)
    print([s.constraint_chk() for s in dataset.system_manager.server.values()])
    print("t: ", dataset.system_manager.total_time())
    print("reward: {:.3f}".format(dataset.system_manager.get_reward(timeslot=0)))
    print("took: {:.3f} sec".format(time.time() - start))
    print("---------- Greedy Algorithm ----------\n")

    # start = time.time()
    # y = heuristic.run_algo()
    # dataset.system_manager.set_y_mat(y)
    # print("---------- Heuristic Algorithm ----------")
    # print("x: ", dataset.system_manager._x)
    # print("y: ", y)
    # print([s.constraint_chk() for s in dataset.system_manager.server.values()])
    # print("t: ", dataset.system_manager.total_time())
    # print("reward: {:.3f}".format(dataset.system_manager.get_reward(timeslot=0)))
    # print("took: {:.3f} sec".format(time.time() - start))
    # print("---------- Heuristic Algorithm ----------\n")

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
    y, v = memetic.run_algo(loop=200, mutation_ratio=0.1, cross_over_ratio=0.1)
    dataset.system_manager.set_y_mat(y)
    print("---------- Memetic Algorithm ----------")
    print("x: ", dataset.system_manager._x)
    print("y: ", y)
    print([s.constraint_chk() for s in dataset.system_manager.server.values()])
    print("t: ", dataset.system_manager.total_time())
    print("reward: {:.3f}".format(dataset.system_manager.get_reward(timeslot=0)))
    print("took: {:.3f} sec".format(time.time() - start))
    print("---------- Memetic Algorithm ----------\n")