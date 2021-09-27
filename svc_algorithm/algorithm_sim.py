import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class Memetic:
    def __init__(self, popularity, edge_cluster, svc_set):
        self.pop_lst = popularity
        self.num_edge = edge_cluster.num_edge
        self.num_micro_service = edge_cluster.num_micro_service
        self.edge_cluster = edge_cluster
        self.svc_set = svc_set

        self.msvc_pop = np.empty(1,dtype=float)  # service -> micro_svc
        self.msvc_deadline = np.empty(1, dtype=float)  # service -> micro_svc
        for svc_idx in range(self.svc_set.num_service):
            msvc_pop = np.full(self.svc_set.svc_lst[svc_idx].num_micro_service,
                               self.pop_lst[svc_idx])
            msvc_deadline = np.full(self.svc_set.svc_lst[svc_idx].num_micro_service,
                                    self.svc_set.svc_lst[svc_idx].deadline)

            self.msvc_pop = np.append(self.msvc_pop, msvc_pop, axis=0)
            self.msvc_deadline = np.append(self.msvc_deadline, msvc_deadline, axis=0)
        self.msvc_pop = np.delete(self.msvc_pop, [0])
        self.msvc_deadline = np.delete(self.msvc_deadline, [0])

        for svc_idx in range(self.svc_set.num_service):
            msvc_pop = np.full(self.svc_set.svc_lst[svc_idx].num_micro_service, self.pop_lst[svc_idx])
            self.msvc_pop = np.append(self.msvc_pop, msvc_pop, axis=0)
        self.msvc_pop = np.delete(self.msvc_pop, [0])

    def generate_random_solutions(self, num_solutions):
        # generate random 3d array
        # return_arr = (num_solution, num_edge, num_micro_service)
        return np.random.randint(0, 2, (num_solutions, self.num_edge, self.num_micro_service))

    def reparation(self, k):
        while True:
            need_fix = np.where(np.logical_not(self.edge_cluster.constraint_chk(k)))[0]
            if len(need_fix) > 0:
                target_edge = np.random.choice(need_fix)
                pushed_msvc = k[target_edge, :]
                # msvc_deadline = pushed_msvc * self.msvc_deadline  # select deadline
                # min_deadline = msvc_deadline[pushed_msvc].min()
                target_deadline = self.msvc_deadline[np.where(pushed_msvc)].max()
                target_msvc = np.where(np.logical_and(self.msvc_deadline == target_deadline, pushed_msvc))[0]
                if len(target_msvc) > 1:
                    target_msvc = np.random.choice(target_msvc)
                else:
                    target_msvc = target_msvc[0]
                # non dynamic solution
                target_svc_idx = self.svc_set.find_svc_idx(target_msvc)
                target_svc = self.svc_set.svc_lst[target_svc_idx]
                offset = self.svc_set.find_k_idx(target_svc_idx, 0)

                '''
                for i in range(target_svc.start, target_svc.end):  # continuous type
                    if K[target_edge, 1 + i + offset]:
                        target_msvc = i + 1
                    else:
                        break
                '''

                target_idx = np.where(pushed_msvc[offset:target_svc.end + offset + 1])[0][-1]

                k[target_edge, offset + target_idx] = 0
                val_host = np.logical_and(self.edge_cluster.avail_map[:, 0] > self.svc_set.require_map[0, target_idx],
                                          self.edge_cluster.avail_map[:, 1] > self.svc_set.require_map[1, target_idx])

                if np.any(val_host):
                    k[np.random.choice(np.where(val_host)[0]), target_idx] = True
            else:
                return k

    '''
    def local_search(self, K):  # todo collect distributed msvc -> one host
        msvc_idx = 0
        for svc_idx in self.svc_set.num_service:
            num_msvc = self.svc_set.svc_lst[svc_idx].num_micro_service
            target = np.logical_xor(K[:, msvc_idx:(msvc_idx+num_msvc - 1)], K[:, msvc_idx+1:(msvc_idx+num_msvc)])
            target_map = np.where(target)
            # for target_idx in range(len(target_map[0])):
                # if

            msvc_idx += self.svc_set.svc_lst[svc_idx].num_micro_service
    '''

    def run_algo(self, num_sol, loop, mutation_ratio=0.005, cross_over_ratio=0.005):
        p_t = self.generate_random_solutions(num_sol)
        for i in range(num_sol):
            p_t[i, :, :] = self.reparation(p_t[i, :, :])

        for _ in range(loop):
            p_t_unknown = np.copy(p_t)
            for _ in range(p_t.shape[0]):
                p = np.random.random(2)
                if p[0] < mutation_ratio:
                    p_t_unknown, a_idx = self.mutation(p_t_unknown)
                    p_t_unknown[a_idx, :, :] = self.reparation(p_t_unknown[a_idx, :, :])
                if p[1] < cross_over_ratio:
                    p_t_unknown, a_idx, b_idx = self.cross_over(p_t_unknown)
                    p_t_unknown[a_idx, :, :] = self.reparation(p_t_unknown[a_idx, :, :])
                    p_t_unknown[b_idx, :, :] = self.reparation(p_t_unknown[b_idx, :, :])
            p_t = self.selection(p_t, p_t_unknown)
        return p_t[0, :, :]

    def selection(self, k_lst_0, k_lst_1):
        ev_0 = self.evaluation(k_lst_0)
        ev_1 = self.evaluation(k_lst_1)
        ev_lst = list()
        ev_lst += list(map(lambda x: (0, x[0], x[1]), enumerate(ev_0)))
        ev_lst += list(map(lambda x: (1, x[0], x[1]), enumerate(ev_1)))
        ev_lst.sort(key=lambda x: x[2], reverse=True)
        result = np.zeros_like(k_lst_0)
        for i in range(result.shape[0]):
            ev = ev_lst[i]
            if ev[0] == 0:
                result[i, :, :] = k_lst_0[ev[1], :, :]
            elif ev[0] == 1:
                result[i, :, :] = k_lst_1[ev[1], :, :]

        return result

    @staticmethod
    def cross_over(k_lst):
        a_idx = np.random.randint(0, k_lst.shape[0])
        while True:
            b_idx = np.random.randint(0, k_lst.shape[0])
            if a_idx != b_idx: break

        a = k_lst[a_idx, :, :]
        b = k_lst[b_idx, :, :]
        k_shape = a.shape
        a = a.ravel()
        b = b.ravel()

        cross_point = np.random.randint(1, a.shape[0] - 1)
        temp = a[:cross_point]
        a[:cross_point] = b[:cross_point]
        b[:cross_point] = temp
        a = a.reshape(k_shape)
        b = b.reshape(k_shape)
        k_lst[a_idx, :, :] = a
        k_lst[b_idx, :, :] = b
        return k_lst, a_idx, b_idx

    @staticmethod
    def mutation(k_lst):
        a_idx = np.random.randint(0, k_lst.shape[0])
        a = k_lst[a_idx, :, :]
        k_shape = a.shape
        a = a.ravel()
        mutation_point = np.random.randint(0, a.shape[0])
        a[mutation_point] = np.logical_not(a[mutation_point])
        a = a.reshape(k_shape)
        k_lst[a_idx, :, :] = a
        return k_lst, a_idx

    def evaluation(self, k_lst):
        evaluation_lst = np.zeros(k_lst.shape[0], dtype=np.float_)
        for k_idx in range(k_lst.shape[0]):
            k = k_lst[k_idx, :, :]
            msvc_idx = 0
            evaluation_lst[k_idx] = 0.0
            for svc_idx in range(self.svc_set.num_service):
                qos = self.pop_lst[svc_idx] / self.svc_set.svc_lst[svc_idx].deadline / self.svc_set.svc_lst[svc_idx].num_micro_service
                for i in range(self.svc_set.svc_lst[svc_idx].num_micro_service):
                    if np.any(k[:, msvc_idx + i]):
                        evaluation_lst[k_idx] += qos
                msvc_idx += self.svc_set.svc_lst[svc_idx].num_micro_service

        return evaluation_lst


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.layer_1 = nn.Linear(input_size, 512)
        self.layer_2 = nn.Linear(512, 512)
        self.layer_3 = nn.Linear(512, 512)
        self.layer_out = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return self.layer_out(x)


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # save transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1)  % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def satisfaction(delay, deadline, alpha):
    if delay < deadline:
        return 1.0
    else:
        return max((alpha * deadline - delay) / ((alpha - 1.0) * deadline), 0.0)


def constraint_chk(action, cluster):
    action.reshape(cluster.K.shape)
    if np.any(np.logical_not(cluster.constraint_chk(action))):
        return 0
    else:
        return 1


def get_state(result_lst, num_item, deadlines, alpha):
    request_arr = np.zeros(num_item, dtype=np.int_)
    satisfaction_sum = 0.0

    for result in result_lst:
        item = result['service_idx']
        delay = result['end'] - result['start']
        satisfaction_sum += satisfaction(delay, deadlines[item], alpha)
        request_arr[item] += 1

    return request_arr, satisfaction_sum


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


class RL:
    def __init__(self, simulator):
        self.model = None
        self.input_size = simulator.get_input_size()
        self.output_size = simulator.get_output_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size=self.input_size, output_size=self.output_size).to(self.device)
        self.target_net = DQN(input_size=self.input_size, output_size=self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.episode_durations = []
        self.sim = simulator

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                result = self.policy_net(state)
                return self.action_translate(state, result)
                # result = result > 0.5
                # return result.view(1, 1, result.size(2))
                # return self.policy_net(state).max()[1].view(1, 1)
                # K, _ = self.action_translate(state, result)
                # return K

        else:
            return torch.tensor([[np.random.randint(0, 2, self.output_size)]], device=self.device, dtype=torch.long)

    def action_translate(self, state, action):
        cpu_res = (0, self.sim.num_edge)
        cpu_req = (cpu_res[1], cpu_res[1] + self.sim.edge_cluster.num_micro_service)
        mem_res = (cpu_req[1], cpu_req[1] + self.sim.num_edge)
        mem_req = (mem_res[1], mem_res[1] + self.sim.edge_cluster.num_micro_service)
        cpu_resource = state[:, :, cpu_res[0]:cpu_res[1]]
        cpu_resource = cpu_resource.view((-1, self.sim.num_edge, 1))
        cpu_resource = cpu_resource.repeat((1, 1, self.sim.edge_cluster.num_micro_service))
        cpu_require = state[:, :, cpu_req[0]:cpu_req[1]]
        cpu_require = cpu_require.repeat((1, self.sim.num_edge, 1))
        mem_resource = state[:, :, cpu_res[0]:cpu_res[1]]
        mem_resource = mem_resource.view((-1, self.sim.num_edge, 1))
        mem_resource = mem_resource.repeat((1, 1, self.sim.edge_cluster.num_micro_service))
        mem_require = state[:, :, mem_req[0]:mem_req[1]]
        mem_require = mem_require.repeat((1, self.sim.num_edge, 1))

        reshaped = action.view((action.shape[0], self.sim.num_edge, self.sim.edge_cluster.num_micro_service))
        K = torch.zeros(reshaped.shape, dtype=torch.bool, device=self.device)  # todo change to gpu
        # reward = torch.zeros(reshaped.shape[0], dtype=torch.float)
        sorted, indices = torch.sort(reshaped, dim=2, descending=True)  # sorting by service  # todo use gather and scatter

        cum_cpu_req = cpu_require.gather(dim=2, index=indices)   # adding resource requirement by reward (descending order)
        cum_cpu_req = cum_cpu_req.cumsum(2)
        cum_mem_req = mem_require.gather(dim=2, index=indices)
        cum_mem_req = cum_mem_req.cumsum(2)

        target = torch.logical_and(cum_cpu_req < cpu_resource, cum_mem_req < mem_resource)
        K.scatter_(2, indices, target)

        K = K.view((reshaped.shape[0], 1, -1))
        return K
        # return K, reward

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
        # 전환합니다.
        batch = Transition(*zip(*transitions))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        # state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        state_action_values = self.policy_net(state_batch) * action_batch
        state_action_values = state_action_values.sum(2).sum(1)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
        # max(1)[0]으로 최고의 보상을 선택하십시오.
        # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).sum(2).sum(1)
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Huber 손실 계산
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, period=60):
        num_episodes = 50
        reward = 0
        for i_episode in range(num_episodes):
            # 환경과 상태 초기화
            self.sim.reset()  # todo simulator reset
            # last_screen = get_screen()
            # current_screen = get_screen()
            # state = current_screen - last_screen
            state = self.sim.get_state()
            state = state.reshape((1, 1, state.size))
            state = np.ascontiguousarray(state, dtype=np.float32)  # nn.linear use float32
            state = torch.from_numpy(state)
            state = state.to(self.device)

            for t in count():
                # 행동 선택과 수행
                action = self.select_action(state)
                self.sim.set_k(np.array(action.cpu().data[0][0]))  # set action
                # _, reward, done, _ = env.step(action.item()) # simulate(action) todo
                reward, done = self.sim.step(period, rl=True)
                reward = torch.tensor([reward], device=self.device)

                # 새로운 상태 관찰
                next_state = self.sim.get_state()  # todo
                next_state = next_state.reshape((1, 1, next_state.size))
                next_state = np.ascontiguousarray(next_state, dtype=np.float32) # nn.linear use float32
                next_state = torch.from_numpy(next_state)
                next_state = next_state.to(self.device)

                # 메모리에 변이 저장
                self.memory.push(state, action, next_state, reward)

                # 다음 상태로 이동
                state = next_state

                # 최적화 한단계 수행(목표 네트워크에서)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    break
            # 목표 네트워크 업데이트, 모든 웨이트와 바이어스 복사
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                if reward is not None:
                    print("episode %s rewords = %s" % (i_episode, reward))

        print('Complete')



