import numpy as np
import math


class Genetic:
    def __init__(self, env, num_solutions):
        self.env = env
        self.system_manager = env.data_set.system_manager
        self.num_solutions = num_solutions
        self.num_services = env.data_set.num_svc
        self.num_containers = env.data_set.num_containers
        self.num_servers = env.data_set.num_servers
        self.epsilon = 1e-3

    @staticmethod
    def generate_random_solutions(self, size):
        return np.random.randn(size)

    def edge_mask(self, x_mat):
        return self.env.get_mask()

    def reparation(self, x, y):   # repair constraint
        epsilon = self.epsilon
        # for i in range(x.shape[0]):
        result_x = np.copy(x)

        while True:
            need_fix_cpu, need_fix_mem, remain_cpu, remain_mem = self.system_manager.constraint_chk(result_x, y, inv_opt=True)

            if np.any(need_fix_mem):
                for batch_idx in range(need_fix_mem.shape[0]):
                    for edge_idx in range(need_fix_mem.shape[1]):
                        if np.any(need_fix_mem[batch_idx, edge_idx, :]):
                            target = np.random.choice(np.where(need_fix_mem[batch_idx, edge_idx, :]))
                            result_x[batch_idx, edge_idx, target] = 0.

            elif np.any(need_fix_cpu):
                before_xy = (result_x, y)
                overed_ratio = -remain_cpu * need_fix_cpu
                overed_ratio /= self._system_manager.edge_cpu_resource.reshape((1, -1, 1))
                overed_ratio += (1.0 + epsilon)
                k = self._system_manager.xy_to_k_mat(result_x, y)
                # indices = np.where(need_fix_cpu)
                # k[indices] = k[indices] / overed_ratio[(indices[0], indices[1], 0)]
                k = np.divide(k, overed_ratio, out=k, where=need_fix_cpu)
                k[np.where(k < epsilon)] = 0.
                result_x, _ = self._system_manager.k_to_xy_mat(k)
            else:
                break

            mask = self.edge_mask(result_x)
            result_x = result_x * mask
        return result_x

    def remove_overload(self, mat):
        arrival = self._system_manager.container_arrival.reshape((1, -1))
        mu = mat / self._system_manager.computing_mat.reshape((1, -1))
        mat[np.where(arrival > mu)] = 0.
        mask = self.edge_mask(mat)
        mat *= mask

    def fill_unused(self, mat_x, mat_y):
        _, _, remain_cpu, remain_mem = self._system_manager.constraint_chk(mat_x, mat_y)

        for svc in self._system_manager.service_set.svc_set:
            indices = svc.part_indices
            min_v = (self._system_manager.container_arrival[indices[0]:indices[1]] *
                     self._system_manager.computing_mat[indices[0]:indices[1]])
            min_v *= 1.1
            for idx in range(indices[0], indices[1]):
                target = remain_cpu[:, mat_y[idx], 0] >= min_v[idx-indices[0]]
                target = np.where(np.logical_and(mat_x[:, idx] < min_v[idx-indices[0]], target))
                remain_cpu[np.where(target), mat_y[idx], 0] += min_v[idx-indices[0]] - mat_x[np.where(target), idx]
                mat_x[np.where(target), idx] = min_v[idx-indices[0]]

    def local_search(self, mat, y):
        self.remove_overload(mat)
        self.fill_unused(mat, y)
        # pass

    def run_algo(self, loop, mutation_ratio=0.05, cross_over_ratio=0.05):
        y_mat = self._system_manager.get_y()
        p_t = self.generate_random_solutions((self.num_container), self.num_solutions, 0, 200)
        p_t = self.reparation(p_t, y_mat)

        p_t_unknown = np.copy(p_t)
        self.local_search(p_t, y_mat)

        for _ in range(loop):
            q_t = self.selection(p_t, p_t_unknown)

            self.mutation(q_t, mutation_ratio)
            self.cross_over(q_t, cross_over_ratio)
            q_t = self.reparation(q_t, y_mat)

            self.local_search(q_t,y_mat)
            p_t_unknown = np.copy(q_t)
            p_t = self.fitness_selection(p_t, q_t)
        return p_t[0, :]

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

        ev_lst = self.evaluation(union)
        ev_lst = list(map(lambda x: (x[0], x[1]), enumerate(ev_lst)))
        ev_lst.sort(key=lambda x: x[1], reverse=True)
        sorted_idx = list(map(lambda x: x[0], ev_lst))
        union = union[sorted_idx[:max_len], :]
        return union

    @staticmethod
    def cross_over(mat, crossover_rate):
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

            cross_point = np.random.randint(1, a.size - 1)
            temp = a[:cross_point]
            a[:cross_point] = b[:cross_point]
            b[:cross_point] = temp
            a = a.reshape(k_shape)
            b = b.reshape(k_shape)
            mat[a_idx, :] = a
            mat[b_idx, :] = b

    @staticmethod
    def mutation(mat, mutation_ratio):
        rand_arr = np.random.rand(*mat.shape)
        rand_arr = rand_arr < mutation_ratio
        mat = ((np.random.rand(*mat.shape) * 100000) - 50000) * rand_arr + mat
        mat[np.where(mat<0.)] = 0.
        # mat = np.logical_xor(mat, rand_arr)

    def evaluation(self, x_lst):
        evaluation_lst = np.zeros(x_lst.shape[0], dtype=np.float_)
        for k_idx in range(x_lst.shape[0]):
            x = x_lst[k_idx, :]
            self._system_manager.set_x_mat(x)
            T_n = self._system_manager.total_time()
            evaluation_lst[k_idx] = self._system_manager.reward(T_n)

        return evaluation_lst