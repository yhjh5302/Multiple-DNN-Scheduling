import numpy as np
import math
from svc_algorithm import data_generator


class Genetic:
    def __init__(self, system_manager, num_solutions, num_svc):
        self._system_manager = system_manager
        self._num_solutions = num_solutions
        self._num_services = num_svc
        self._num_container = len(self._system_manager.service_set.container_set)
        self._num_edge = len(self._system_manager.edge)
        self._epsilon = 1e-3

    @staticmethod
    def generate_random_solutions(arr_size, num_sol, min_v=0, max_v=1, dtype=np.float_):
        if dtype == np.float_:
            return np.random.random((num_sol, arr_size)) * (max_v - min_v) + min_v
        else:
            return np.random.randint(min_v, max_v+1, arr_size)

    def edge_mask(self, x_mat):
        edge_mask_mat = np.zeros(x_mat.shape, dtype=np.bool_)
        for svc in self._system_manager.service_set.svc_set:
            indices = svc.part_indices
            edge_mask_mat[:, indices[0]:indices[1]] = x_mat[:, indices[0]:indices[1]] > self._epsilon
            edge_mask_mat[:, indices[0]:indices[1]] = np.cumprod(edge_mask_mat[:, indices[0]:indices[1]], axis=1)
        return edge_mask_mat

    def reparation(self, x, y):   # repair constraint
        epsilon = self._epsilon
        # for i in range(x.shape[0]):
        result_x = np.copy(x)

        while True:
            need_fix_cpu, need_fix_mem, remain_cpu, remain_mem = self._system_manager.constraint_chk(result_x, y, inv_opt=True)

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
        # self.fill_unused(mat, y)

    def run_algo(self, loop, mutation_ratio=0.05, cross_over_ratio=0.05):
        # p_t = self.generate_random_solutions(num_sol)
        y_mat = self._system_manager.get_y()
        p_t = self.generate_random_solutions(self._num_container, self._num_solutions, 0, 200)
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

    def calc_utility(self, T_n):
        U_n = np.zeros(shape=(self._num_services, ))
        for n, svc in enumerate(self._system_manager.service_set.svc_set):
            T_n_hat = svc.deadline
            alpha = 2
            if T_n[n] < T_n_hat:
                U_n[n] = 1
            elif T_n_hat <= T_n[n] and T_n[n] < alpha * T_n_hat:
                U_n[n] = 1 - (T_n[n] - T_n_hat) / ((alpha - 1) * T_n_hat)
            else:
                U_n[n] = 0
        return U_n

    def evaluation(self, x_lst):
        evaluation_lst = np.zeros(x_lst.shape[0], dtype=np.float_)
        for k_idx in range(x_lst.shape[0]):
            x = x_lst[k_idx, :]
            self._system_manager.set_x_mat(x)
            self._system_manager.update_edge_computing_time()
            T_n = self._system_manager.total_time()
            U_n = self.calc_utility(T_n)
            utility_factor = 0
            for n in range(self._num_services):
                utility_factor += self._system_manager.service_arrival[n] * U_n[n]

            energy_factor = np.inf
            for d in self._system_manager.edge:
                E_d = d.energy_consumption()
                E_d_hat = d._energy
                energy_factor = min(energy_factor, E_d_hat / E_d)

            reward = energy_factor * utility_factor
            evaluation_lst[k_idx] = reward

        return evaluation_lst


def main():
    d = data_generator.DataSet()
    g = Genetic(d.system_manager, 100, d.num_services)
    x = g.run_algo(100)
    d.system_manager.set_x_mat(x)
    d.system_manager.update_edge_computing_time()
    time = d.system_manager.total_time()
    print("t: ", time)
    reward = g.evaluation(np.array([x]).reshape((1, -1)))
    print("x: ", x)
    print("reward: ", reward[0])
    print("normed reward: ", reward[0])


if __name__=="__main__":
    main()
