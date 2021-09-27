# file for refactoring
# currently useless file
import numpy as np
import random
from collections import deque
import math


COMP_RATIO = 50*(10**6)   # computation per input data  (50 MFLOPS)


class NetworkManager:  # managing data transfer
    def __init__(self, n_0, channel_b, gain):
        self.B_gw = None
        self.B_fog = None
        self.B_cl = None
        self.B_dd = None
        self.P_dd = None
        self.N_0 = n_0
        self.W = channel_b
        self.h = gain

    def communication(self, amount, send, receive):
        if receive == -1:  # fog
            return amount / self.B_fog
        elif receive == -2:  # cloud
            return amount / self.B_cl
        elif receive == -3:  # gateway
            return amount / self.B_gw
        else:
            return amount / self.B_dd[send, receive]

    def cal_b_dd(self):
        self.B_dd = np.zeros_like(self.P_dd)
        for i in range(self.B_dd.shape[0]):
            for j in range(i + 1, self.B_dd.shape[1]):
                self.B_dd[i, j] = self.B_dd[j, i] = self.W * math.log2(1+(self.h * self.P_dd[i, j]/self.N_0))
            self.B_dd[i, i] = float("inf")


class SystemManager:
    def __init__(self):
        self._x = None
        self._y = None
        # self.y_mat = None
        self.service_set = None
        self.edge = None
        self.fog = None
        self.cloud = None
        self.net_manager = None
        self.edge_computing_time_mat = None
        self.mu_mat = None
        self.computing_mat = None   # computing requirement
        self.service_arrival = None  # JIN added
        self.container_arrival = None
        self.transmission_power_mat = None   # E^{tr}_{n,k,d} mat
        self.transmission_time_mat = None  # ep 4 left
        self.probability_mat = None  # P_{n,k}(X,Y) eq 8
        self.inp_data_size = None
        self.edge_cpu_resource = None
        self.edge_mem_resource = None
        self.mem_size = None
        self.edge_mask_mat = None
        self.C = 10    # queue length
        self.high_layer_computing_time_mat = None
        self.high_layer_transmit_time_mat = None

    def set_service_set(self, service_set):
        self.service_set = service_set
        self.inp_data_size = np.zeros(len(self.service_set.container_set), dtype=np.float_)    # data size
        self.computing_mat = np.zeros(len(self.service_set.container_set), dtype=np.float_)  # computing amount
        self.mem_size = np.zeros(len(self.service_set.container_set), dtype=np.int_)
        self.container_arrival = np.zeros(len(self.service_set.container_set), dtype=np.float_)
        self.probability_mat = np.zeros(len(self.service_set.container_set), dtype=np.float_)
        self.reset_xy_mat()
        for container in self.service_set.container_set:
            self.inp_data_size[container.id] = container.inp_data_size
            self.computing_mat[container.id] = container.comp_amount
            self.mem_size[container.id] = container.memory

    def get_state(self, x):
        self.set_x_mat(x)
        state = np.append(self._x.flatten(), self._y.flatten())
        state = np.append(state, (self.edge_cpu_resource - [e.used_cpu for e in self.edge]).flatten())
        state = np.append(state, (self.edge_mem_resource - [e.used_mem for e in self.edge]).flatten())
        state = np.append(state, (self.get_battery() - [e.energy_consumption() for e in self.edge]).flatten())
        return state

    def get_y(self):
        return self._y.flatten()

    def get_battery(self):
        result = np.zeros(len(self.edge), dtype=np.float_)
        for e in self.edge:
            result[e.id] = e.get_energy()
        return result

    def battery_update(self):   # update battery level
        for e in self.edge:
            e.energy_update()

    def set_servers(self, edge, fog, cloud):
        self.edge = edge
        self.fog = fog
        self.cloud = cloud
        self.edge_mem_resource = np.zeros(len(self.edge), dtype=np.int_)
        self.edge_cpu_resource = np.zeros(len(self.edge), dtype=np.float_)
        for id, e in enumerate(self.edge):
            e.id = id
            self.edge_mem_resource[id] = e.memory
            self.edge_cpu_resource[id] = e.cpu

        self.fog.id = -1
        self.cloud.id = -2

    def init_servers(self):
        target_svc = np.random.randint(0, 2, len(self.service_set.svc_set))
        total_container = 0
        for svc in self.service_set.svc_set:
            total_container += len(svc.partitions)

        for svc in self.service_set.svc_set:
            if target_svc[svc.id]:
                for i in range(svc.part_indices[0], svc.part_indices[1]):
                    self.fog.deployed_container[i] = self.service_set.container_set[i]
                    self.fog.deployed_container_cpu[i] = self.fog.cpu / total_container
                    self.fog.deployed_container_mem[i] = self.mem_size[i]
            for i in range(svc.part_indices[0], svc.part_indices[1]):
                self.cloud.deployed_container[i] = self.service_set.container_set[i]
                self.cloud.deployed_container_cpu[i] = self.cloud.cpu
                self.cloud.deployed_container_mem[i] = self.mem_size[i]

    def calculate_high_layer(self):
        self.high_layer_computing_time_mat = np.zeros(len(self.service_set.container_set), dtype=np.float_)
        self.high_layer_transmit_time_mat = np.zeros(len(self.service_set.container_set), dtype=np.float_)
        for container in self.service_set.container_set:
            if container.id in self.fog.deployed_container:
                self.high_layer_computing_time_mat[container.id] = self.fog.computing_time(container)
                self.high_layer_transmit_time_mat[container.id] = self.net_manager.communication(container.inp_data_size, 0, -1)
            else:
                self.high_layer_computing_time_mat[container.id] = self.cloud.computing_time(container)
                self.high_layer_transmit_time_mat[container.id] = self.net_manager.communication(container.inp_data_size, 0, -2)

        for svc in self.service_set.svc_set:
            self.high_layer_computing_time_mat[svc.part_indices[1]:svc.part_indices[0]] = \
                self.high_layer_computing_time_mat[svc.part_indices[1]:svc.part_indices[0]].cumsum()

    def reset_xy_mat(self):
        self._x = np.zeros(len(self.service_set.container_set), dtype=np.float_)
        self._y = np.full(len(self.service_set.container_set), -1, dtype=np.int_)
        self.edge_mask_mat = None

    def set_x_mat(self, mat_x):
        self._x = mat_x
        for c_id, s_id in enumerate(self._y):
            if s_id > -1:
                self.edge[int(s_id)].deploy_one(self.service_set.container_set[c_id], self._x[c_id])
            elif s_id == -1:
                self.fog.deploy_one(self.service_set.container_set[c_id], self._x[c_id])
            elif s_id == -2:
                self.cloud.deploy_one(self.service_set.container_set[c_id], self._x[c_id])
        self.edge_mask()

    def set_y_mat(self, mat_y):
        self._y = mat_y
        for c_id, s_id in enumerate(self._y):
            if s_id > -1:
                self.edge[int(s_id)].deploy_one(self.service_set.container_set[c_id], 0.)
            elif s_id == -1:
                self.fog.deploy_one(self.service_set.container_set[c_id], 0.)
            elif s_id == -2:
                self.cloud.deploy_one(self.service_set.container_set[c_id], 0.)
        self.edge_mask()

    def set_xy(self, c_id, x, y):
        self._x[c_id] = x
        self._y[c_id] = y
        self.edge[self._y[c_id]].deploy_one(self.service_set.container_set[c_id], x)
        self.edge_mask()

    def set_arrival_rate(self, arrival):
        self.service_arrival = arrival
        for svc in self.service_set.svc_set:
            self.container_arrival[svc.part_indices[0]:svc.part_indices[1]] = arrival[svc.id]

    def front_uninstalled_partition_mask(self, installed_mask):
        if len(installed_mask.shape) > 1:
            installed_mask = installed_mask[0, :]
        mask = np.zeros((1, len(self.service_set.container_set)), dtype=np.bool_)
        for svc in self.service_set.svc_set:
            indices = svc.part_indices
            # continue_mask = np.cumprod(installed_mask[:, indices[0]:indices[1]])
            # mask[:,indices[0]:indices[1]] = continue_mask
            target = self.front_uninstalled_partition(installed_mask, svc)
            if target < indices[1]:
                mask[:, target] = True

        return mask

    @staticmethod
    def front_uninstalled_partition(installed_mask, svc):
        if len(installed_mask.shape) > 1:
            installed_mask = installed_mask[0, :]
        indices = svc.part_indices
        continue_mask = np.cumprod(installed_mask[indices[0]:indices[1]])
        return continue_mask.sum() + indices[0]

    def calculate_future_t_tr(self, y):
        result = np.zeros((len(self.edge), len(self.service_set.container_set)), dtype=np.float_)
        installed_mask = y > -1
        for svc in self.service_set.svc_set:
            target_idx = self.front_uninstalled_partition(installed_mask, svc)
            if target_idx < svc.part_indices[1]:
                target_c = self.service_set.container_set[target_idx]
                if target_idx == svc.part_indices[0]:
                    result[:, target_idx] = self.net_manager.communication(target_c.inp_data_size, 0, -3)
                else:
                    ex_y = y[0, target_idx - 1]
                    for e in self.edge:
                        result[e.id, target_idx] = self.net_manager.communication(target_c.inp_data_size, ex_y, e.id)
        return result

    def update_transmission_mats(self):
        self.transmission_power_mat = np.zeros((len(self.edge), len(self.service_set.container_set)), dtype=np.float_)
        self.transmission_time_mat = np.zeros((len(self.edge), len(self.service_set.container_set)), dtype=np.float_)
        data_size = self.inp_data_size * self.container_arrival

        for svc in self.service_set.svc_set:
            if self._y[svc.part_indices[0]] >= 0:  # start in edge
                ex_power_val = 0.
                ex_time_val = 0.
                for i in range(svc.part_indices[0], svc.part_indices[1] - 1):
                    if self._y[i] >= 0:  # deployed in edge
                        # transfer to next
                        self.transmission_time_mat[self._y[i], i] = \
                            self.net_manager.communication(data_size[i+1], self._y[i], self._y[i + 1]) + ex_time_val
                        ex_time_val = self.transmission_time_mat[self._y[i], i]
                        self.transmission_power_mat[self._y[i], i] = \
                            self.net_manager.communication(data_size[i+1], self._y[i], self._y[i + 1]) * \
                            self.net_manager.P_dd[self._y[i], self._y[i + 1]] + ex_power_val
                        ex_power_val = self.transmission_power_mat[self._y[i], i]

    def update_edge_computing_time(self):  # compute queuing time & drop probability
        edge_mask = self._x > 0.
        self.mu_mat = self._x / self.computing_mat
        mu_inv = np.divide(1, self.mu_mat, out=np.zeros_like(self.mu_mat), where=self.mu_mat != 0.)

        rho = self.container_arrival * mu_inv   # lambda / mu
        rho_mask = rho >= 1.0
        rho[np.where(rho_mask)] = 0.0
        edge_mask = np.logical_and(edge_mask, np.logical_not(rho_mask))

        divisor = 2 * (1. - rho)
        waiting = np.divide(self.container_arrival * mu_inv, divisor, out=np.zeros_like(divisor),
                            where=divisor != 0.)

        mask = np.zeros(len(self.mu_mat), dtype=np.bool_)

        pi = np.zeros_like(rho)   # calculate block probability
        for k in range(1, self.C):
            pi += pow(math.e, k * rho) * pow(-1, (self.C - k)) * (pow(k * rho, self.C - k) / math.factorial(self.C - k) + pow(k * rho, self.C - k - 1) / math.factorial(self.C - k - 1))
        pi += pow(math.e, self.C * rho)
        pi *= 1 - rho
        """
        fac_1 = 1
        for k in range(self.C - 1, 0, -1):
            fac_2 = fac_1
            fac_1 *= self.C - k
            pi += (math.e ** (k * rho)) * ((-1) ** (self.C - k)) * ((k * rho) ** (self.C - k) / fac_1 +
                                                                    (k * rho) ** (self.C - k - 1) / fac_2)

        pi += math.e ** (self.C * rho)
        pi *= (1 - rho)
        """
        for svc in self.service_set.svc_set:
            indices = svc.part_indices
            mask[indices[0]] = True
            mask[indices[0]+1:indices[1]] = \
                self.mu_mat[indices[0]:indices[1] - 1] > self.mu_mat[indices[0]+1:indices[1]]  # eq 2

            self.probability_mat[indices[0]:indices[1]] = 1. - pi[indices[0]:indices[1]]
            self.probability_mat[indices[0]:indices[1]] = (self.probability_mat[indices[0]:indices[1]] *
                                                           edge_mask[indices[0]:indices[1]])
            self.probability_mat[indices[0]:indices[1]] = self.probability_mat[indices[0]:indices[1]].cumprod()

        self.edge_computing_time_mat = mu_inv + (waiting * mask)
        self.edge_computing_time_mat = self.edge_computing_time_mat * self.edge_mask()

    def start_time_cal(self, svc):
        first_idx = svc.part_indices[0]
        if self._y[first_idx] >= 0:
            return self.net_manager.communication(svc.partitions[0].inp_data_size, 0, -3)  # to gateway
        else:
            return self.net_manager.communication(svc.partitions[0].inp_data_size, 0, self._y[first_idx])  # fog or cloud

    def total_time(self):  # calculate eq 10
        result = np.zeros(len(self.service_set.svc_set), dtype=np.float_)
        for svc in self.service_set.svc_set:
            indices = svc.part_indices
            result[svc.id] = self.start_time_cal(svc)
            result[svc.id] += (self.high_layer_computing_time_mat[indices[0]] +
                              self.high_layer_transmit_time_mat[indices[0]]) * (1 - self.probability_mat[indices[0]])

            result[svc.id] += np.sum(
                self.probability_mat[indices[0]:indices[1]-1] * (self.edge_computing_time_mat[indices[0]:indices[1]-1] +
                    self.high_layer_computing_time_mat[indices[0]+1:indices[1]] +
                    self.high_layer_transmit_time_mat[indices[0]+1:indices[1]]
                )
            )
            result[svc.id] += self.probability_mat[indices[1]-1] * self.edge_computing_time_mat[indices[1]-1]

        return result

    def edge_mask(self):
        self.edge_mask_mat = np.zeros(len(self.service_set.container_set), dtype=np.bool_)
        for svc in self.service_set.svc_set:
            indices = svc.part_indices
            self.edge_mask_mat[indices[0]:indices[1]] = self._y[indices[0]:indices[1]] >= 0
            self.edge_mask_mat[indices[0]:indices[1]] = np.cumprod(self.edge_mask_mat[indices[0]:indices[1]])

        return self.edge_mask_mat

    def constraint_chk(self, mat_x, mat_y=None, inv_opt=False):
        if mat_y is None:
            mat_y = self._y
        mat_k = self.xy_to_k_mat(mat_x, mat_y)
        deploy_mat = mat_k > 0.

        used_cpu = np.sum(mat_k, axis=2, keepdims=True)   # (batch, edge, container)
        used_mem = np.sum(deploy_mat * self.mem_size, axis=2, keepdims=True)
        remain_cpu = self.edge_cpu_resource.reshape((1, -1, 1)) - used_cpu
        remain_mem = self.edge_mem_resource.reshape((1, -1, 1)) - used_mem

        # chk = np.logical_and(remain_cpu >= 0., remain_mem >= 0.)
        # if inv_opt:
        #     chk = np.logical_not(chk)
        if inv_opt:
            return remain_cpu < 0., remain_mem < 0., remain_cpu, remain_mem
        else:
            return remain_cpu >= 0., remain_mem >= 0., remain_cpu, remain_mem

    def xy_to_k_mat(self, mat_x, mat_y):
        mat_y = np.copy(mat_y)
        result = np.zeros((mat_x.shape[0], len(self.edge), len(self.service_set.container_set)), dtype=np.float_)
        if len(mat_y.shape) != len(mat_x.shape):
            mat_y = mat_y.reshape((1, 1, -1))
        if mat_y.shape[0] != mat_x.shape[0]:
            mat_y = mat_y.repeat(mat_x.shape[0], axis=0)

        indices = np.where(mat_y >= 0)
        result[indices[0], mat_y[indices], indices[2]] = mat_x[indices[0], indices[2]]
        return result

    @staticmethod
    def k_to_xy_mat(k):
        mat_y = np.zeros((k.shape[0], k.shape[2]), dtype=np.int_)
        mat_x = np.zeros((k.shape[0], k.shape[2]), dtype=np.float_)
        indices = np.where(k > 0.)
        mat_y[(indices[0], indices[2])] = indices[1]
        mat_x[(indices[0], indices[2])] = k[indices]

        return mat_x, mat_y


class ServiceSet:
    #  set of services in simulation
    def __init__(self):
        self.svc_set = list()   # the set of services
        self.container_set = list()   # the set of whole containers

    def add_services(self, svc):
        svc.id = len(self.svc_set)
        self.svc_set.append(svc)

        start_id = len(self.container_set)
        for partition in svc.partitions:
            partition.id = len(self.container_set)
            self.container_set.append(partition)
        end_id = len(self.container_set)
        svc.part_indices = (start_id, end_id)


class Service:
    #  the one of services in simulation
    def __init__(self, deadline): # JIN added
        self.id = None
        self.partitions = list()
        self.part_indices = None
        self.deadline = deadline # JIN added

    def create_partitions(self, op_type=0, *opt):
        if op_type is 0:   # input is just number of partitions
            for i in range(opt[0]):
                mem = random.randint(4, 50 * 1024)  # 4KB~50MB
                inp_data_size = random.randint(4, 1024)
                partition = Container(mem, inp_data_size, inp_data_size * COMP_RATIO)
                self.partitions.append(partition)

        elif op_type is 1:  # input is list of partition options  (mem, input_size)
            for options in opt[0]:
                partition = Container(options[0], options[1], options[1] * COMP_RATIO)
                self.partitions.append(partition)
        else:
            pass


class Container:
    # the partition of the service in simulation
    def __init__(self, mem, inp_data_size, comp_amount):
        self.id = None
        self.memory = mem
        self.inp_data_size = inp_data_size
        self.comp_amount = comp_amount
        self._x = None
        self._y = None
        self._fog = None
        self._cloud = None

    def set_fog_n_cloud(self, fog, cloud):
        self._fog = fog
        self._cloud = cloud

    def reset_xy(self):
        self._x = None
        self._y = None

    """    
        def update_xy(self, y=None):
            if y is not None:
                self._y = y
    
            self._x = self._y.deployed_container[self.id] # JIN need to modify
    
        def undeploy(self):
            if self._fog.deployed_container[self.id]:
                self.update_xy(self._fog)
            else:
                self.update_xy(self._cloud)
    """
    def update_xy(self, x=None, y=None):
        self._y = y
        self._x = x

    def computing_time(self):
        if self._x is not None and self._x > 0:
            return self.comp_amount / self._x
        else:
            return 0.

    def service_rate(self):
        return self._x / self.comp_amount


class Server:
    #  elements with computing power
    def __init__(self, cpu, memory, ipc, **kwargs):
        self.id = None
        self.cpu = cpu * ipc
        self.cpu_clk = cpu
        self.memory = memory
        self.ipc = ipc
        self.deployed_container = dict()
        self.deployed_container_cpu = dict()
        self.deployed_container_mem = dict()
        self.used_cpu = 0.
        self.used_mem = 0.
        self.svc_set = None

        for key, value in kwargs.items():
            setattr(self, key, value)
        self.pushed_lst = deque()

    def computing_time(self, container):
        cpu = self.deployed_container_cpu.get(container.id)
        if cpu and cpu > 0.:
            return container.comp_amount / self.deployed_container_cpu[container.id]
        else:
            return -1.0

    def reset(self):
        self.used_cpu = self.used_mem = 0
        self.pushed_lst.clear()
        while self.deployed_container:
            _, container = self.deployed_container.popitem()
            container.reset_xy()
        if self.deployed_container_cpu:
            self.deployed_container_cpu.clear()
        if self.deployed_container_mem:
            self.deployed_container_mem.clear()

    def deploy_one(self, container, computing_resource):
        self.deployed_container[container.id] = container
        self.deployed_container_cpu[container.id] = computing_resource
        self.deployed_container_mem[container.id] = container.memory
        self.used_cpu += computing_resource
        self.used_mem += container.memory
        container.update_xy(computing_resource, self.id)

    def constraint_chk(self, *args):
        if self.used_cpu <= self.cpu and self.used_mem <= self.memory:
            return True
        else:
            return False


class Edge(Server):
    def __init__(self, cpu, memory, ipc, **kwargs):
        super().__init__(cpu, memory, ipc, **kwargs)
        if not hasattr(self, "_energy"):  # battery
            self._energy = 100.

        if not hasattr(self, "_tau"):   # timeslot
            self._tau = 60. * 60.    # 1hour default

        if not hasattr(self, "_capacitance"):  # capacitance value (need reference)
            self._capacitance = 1e-6

        if not hasattr(self, "system_manager"):
            self.system_manager = None

        if not hasattr(self, "min_energy"):
            self.min_energy = 1.0

        if not hasattr(self, "max_energy"):
            self.max_energy = 10.0

        self.__trans_energy_ratio = 10.  # need reference

    def constraint_chk(self):
        if self._energy <= self.energy_consumption():
            return False
        else:
            return super().constraint_chk()

    def energy_update(self):
        self._energy -= self.energy_consumption()
        if self._energy <= 0.:
            self._energy = 0.
            self.reset()

    def get_energy(self):
        return self._energy

    def energy_consumption(self):
        result = self.computation_energy_consumption() + self.transmission_energy_consumption()
        return result

    def computation_energy_consumption(self):
        return self.min_energy + (self.max_energy - self.min_energy) * self.used_cpu / self.cpu

    def transmission_energy_consumption(self):
        result = 0.
        for svc in self.system_manager.service_set.svc_set:
            indices = svc.part_indices
            result += np.sum(
                self.system_manager.transmission_power_mat[self.id, indices[0]:indices[1]] *
                self.system_manager.probability_mat[indices[0]:indices[1]]
            )
        return result


def spd_algorithm(systemmanager, beta_c, beta_t, beta_b):
    # omaga = np.zeros((len(systemmanager.service_set.container_set), len(systemmanager.edge)), dtype=np.float_)
    edge_cpu = np.zeros((len(systemmanager.edge), 1), dtype=np.float_)
    used_cpu = np.zeros_like(edge_cpu)
    edge_mem = np.zeros((len(systemmanager.edge), 1), dtype=np.int_)
    used_mem = np.zeros_like(edge_mem)
    edge_battery = np.zeros((len(systemmanager.edge), 1), dtype=np.float_)
    energy_min = np.zeros((len(systemmanager.edge), 1), dtype=np.float_)
    energy_increase = np.zeros((len(systemmanager.edge), 1), dtype=np.float_) # max - min

    r_cpu = np.zeros((1, len(systemmanager.service_set.container_set)), dtype=np.float_)
    r_mem = np.zeros((1, len(systemmanager.service_set.container_set)), dtype=np.int_)

    for e in systemmanager.edge:
        edge_cpu[e.id, 0] = e.cpu
        edge_mem[e.id, 0] = e.memory
        edge_battery[e.id, 0] = e.get_energy()
        energy_min[e.id, 0] = e.min_energy
        energy_increase[e.id, 0] = e.max_energy - e.min_energy

    for c in systemmanager.service_set.container_set:
        r_cpu[0, c.id] = c.comp_amount
        r_mem[0, c.id] = c.memory

    for svc in systemmanager.service_set.svc_set:
        r_cpu[0, svc.part_indices[0]:svc.part_indices[1]] /= (svc.deadline * len(svc.partitions))

    result = np.full((1, r_cpu.size), -1, dtype=np.int_)   # 2d array

    while -1 in result:
        mask = result == -1
        mask = mask.repeat(edge_cpu.size, axis=0)
        target = systemmanager.front_uninstalled_partition_mask(result > -1)
        mask = np.logical_and(mask, target)
        mask = np.logical_and(
            mask, used_mem + r_mem < edge_mem
        )

        if not np.any(mask):
            break
        # f_cpu = r_cpu + used_cpu
        u_cpu = np.divide(r_cpu + used_cpu, edge_cpu, out=np.full((edge_cpu.size, r_cpu.size), float("inf"),
                          dtype=np.float_), where=edge_cpu != 0.)
        u_tr = systemmanager.calculate_future_t_tr(result)
        u_b = np.divide(energy_increase * u_cpu, edge_battery, out=np.full((edge_cpu.size, r_cpu.size), float("inf"),
                        dtype=np.float_), where=edge_battery != 0.)

        w = beta_c*u_cpu + beta_t * u_tr + beta_b * u_b
        w = np.ma.masked_array(w, np.logical_not(mask))
        min_w = w.argmin()
        min_w = np.unravel_index(min_w, w.shape)
        result[0, min_w[1]] = min_w[0]

        used_cpu[min_w[0], 0] += r_cpu[0, min_w[1]]
        used_mem[min_w[0], 0] += r_mem[0, min_w[1]]
        # mask = np.logical_and(mask, used_mem+r_mem < edge_mem)

    return result.flatten()