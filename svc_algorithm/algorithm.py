import math
import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
from config import *

node_axis = 0
msvc_axis = 1


class FrontPart:
    def __init__(self):
        self.arrival = None
        self.blabla = None

    def calculate_util(self):
        pass


class Memetic:
    def __init__(self, popularity, simple_sim):
        self.pop_lst = popularity    #
        self.num_edge = len(simple_sim.node_info)
        self.num_micro_service = len(simple_sim.msvc_data)
        self.simulator = simple_sim

        # self.msvc_pop = np.empty(1, dtype=float)  # service -> micro_svc
        # self.msvc_deadline = np.empty(1, dtype=float)  # service -> micro_svc
        self.msvc_pop = np.zeros(self.num_micro_service, dtype=np.float_)
        self.msvc_deadline = np.zeros(self.num_micro_service, dtype=np.float_)
        self.reverse_svc_map = dict()

        for svc_key in self.simulator.svc_info:
            msvc_map = self.simulator.svc_map[svc_key]
            self.msvc_pop[self.simulator.key_to_idx(msvc_map)] = self.pop_lst[self.simulator.key_to_idx(svc_key)]
            self.msvc_deadline[self.simulator.key_to_idx(msvc_map)] = self.simulator.svc_info[svc_key][1]
            for msvc_key in msvc_map:
                self.reverse_svc_map[msvc_key] = svc_key

    def generate_random_solutions(self, num_solutions):
        # generate random 3d array
        # return_arr = (num_solution, num_edge, num_micro_service)
        return np.random.randint(2, size=(num_solutions, self.num_edge, self.num_micro_service), dtype=np.bool_)

    def reparation(self, k):
        for i in range(k.shape[0]):
            while True:
                # need_fix = np.where(np.logical_not(self.edge_cluster.con
                # straint_chk(k)))[0]
                need_fix = self.simulator.constraint_chk(k[i, :, :], result_arr=True)
                need_fix = np.where(np.logical_not(need_fix))[0]
                if len(need_fix) > 0:
                    target_edge = np.random.choice(need_fix)
                    pushed_msvc = k[i, target_edge, :]
                    # msvc_deadline = pushed_msvc * self.msvc_deadline  # select deadline
                    # min_deadline = msvc_deadline[pushed_msvc].min()
                    target_deadline = self.msvc_deadline[np.where(pushed_msvc)].max()
                    target_msvc = np.where(np.logical_and(self.msvc_deadline == target_deadline, pushed_msvc))[0]
                    if len(target_msvc) > 1:
                        target_msvc = np.random.choice(target_msvc)
                    else:
                        target_msvc = target_msvc[0]
                    # non dynamic solution
                    target_svc_key = self.reverse_svc_map[self.simulator.idx_to_key(target_msvc)]
                    offset = self.simulator.svc_map[target_svc_key][0]
                    offset = self.simulator.key_to_idx(offset)
                    offset_end = self.simulator.svc_map[target_svc_key][-1]
                    offset_end = self.simulator.key_to_idx(offset_end)

                    target_idx = np.where(pushed_msvc[offset:offset_end + 1])[0][-1]
                    k[i, target_edge, offset + target_idx] = False

                    avail_map = self.simulator.avail_map(k[i, :, :])
                    t_require_map = self.simulator.require_map[:, target_idx].reshape((1, 2))
                    # val_host = np.logical_and(avail_map[:, 0] > self.simulator.require_map[0, target_idx],
                    #                           avail_map[:, 1] > self.simulator.require_map[1, target_idx])
                    val_host = np.all(avail_map >= t_require_map, axis=1)

                    if np.any(val_host):
                        k[i, np.random.choice(np.where(val_host)[0]), target_idx] = True
                else:
                    break

    def local_search(self, mat):
        for i in range(mat.shape[0]):
            for svc_key in self.simulator.svc_map:
                msvc_idx = self.simulator.svc_map[svc_key]
                msvc_idx = self.simulator.key_to_idx(msvc_idx)

                mat[i, :, msvc_idx] = self.remove_discontinuity(mat[i, :, msvc_idx])  # remove useless item(=remove discontinuity)
                mat[i, :, msvc_idx] = self.remove_overlap(mat[i, :, msvc_idx])  # remove overlapped item
            self.fill_to_limit(mat[i, :, :])

    def fill_to_limit(self, mat):  # fill using random
        target_msvc = np.logical_not(np.any(mat, axis=node_axis))
        target_svc = dict()
        for msvc_idx in np.where(target_msvc)[0]:
            msvc_key = self.simulator.idx_to_key(msvc_idx)
            svc_key = self.reverse_svc_map[msvc_key]
            if svc_key not in target_svc:
                target_svc[svc_key] = list()
            target_svc[svc_key].append(msvc_key)

        for svc_key in target_svc:
            target_svc[svc_key].sort()

        avail_map = self.simulator.avail_map(mat)  # init avail_map
        target_svc_lst = list(target_svc.keys())
        while target_svc:
            target_svc_key = np.random.choice(target_svc_lst)
            target_msvc_key = target_svc[target_svc_key].pop()
            target_msvc_idx = self.simulator.key_to_idx(target_msvc_key)
            resource_require = self.simulator.require_map[:, target_msvc_idx]
            resource_require = resource_require.reshape((1, 2))
            target_node_lst = np.where(np.all(avail_map >= resource_require, axis=1))[0]

            if target_node_lst.size > 0:
                if not target_svc[target_svc_key]:  # remove empty
                    del target_svc[target_svc_key]
                    target_svc_lst.remove(target_svc_key)
                target_node_idx = np.random.choice(target_node_lst)
                mat[target_node_idx, target_msvc_idx] = True
                avail_map = self.simulator.avail_map(mat)

            else:
                del target_svc[target_svc_key]  # remove big svc
                target_svc_lst.remove(target_svc_key)

    @staticmethod
    def remove_discontinuity(mat):
        in_fog = mat.any(axis=node_axis)
        in_fog = in_fog.reshape((1, -1))
        return mat * in_fog.cumprod(axis=msvc_axis)

    @staticmethod
    def remove_overlap(mat):
        num_deploy = mat.sum(axis=node_axis)
        overlapped = np.where(num_deploy > 1)[0]   # overlapped index
        for idx in overlapped:
            target = np.where(mat[:, idx])[0]
            mat[:, idx] = False
            mat[np.random.choice(target), idx] = True   # random choice one

    def run_algo(self, num_sol, loop, mutation_ratio=0.05, cross_over_ratio=0.05):
        p_t = self.generate_random_solutions(num_sol)
        self.reparation(p_t)

        p_t_unknown = np.copy(p_t)
        self.local_search(p_t)

        for _ in range(loop):
            q_t = self.selection(p_t, p_t_unknown)

            self.mutation(q_t, mutation_ratio)
            self.cross_over(q_t, cross_over_ratio)
            self.reparation(q_t)

            self.local_search(q_t)
            p_t_unknown = np.copy(q_t)
            p_t = self.fitness_selection(p_t, q_t)
        return p_t[0, :, :]

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
        return union[order[:max_len], :, :]

    def fitness_selection(self, *args):
        union, max_len = self.union(*args)

        ev_lst = self.evaluation(union)
        ev_lst = list(map(lambda x: (x[0], x[1]), enumerate(ev_lst)))
        ev_lst.sort(key=lambda x: x[1], reverse=True)
        sorted_idx = list(map(lambda x: x[0], ev_lst))
        union = union[sorted_idx[:max_len], :, :]
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

            a = mat[a_idx, :, :]
            b = mat[b_idx, :, :]
            k_shape = a.shape

            a = a.ravel()  # flatten
            b = b.ravel()

            cross_point = np.random.randint(1, a.size - 1)
            temp = a[:cross_point]
            a[:cross_point] = b[:cross_point]
            b[:cross_point] = temp
            a = a.reshape(k_shape)
            b = b.reshape(k_shape)
            mat[a_idx, :, :] = a
            mat[b_idx, :, :] = b

    @staticmethod
    def mutation(mat, mutation_ratio):
        rand_arr = np.random.rand(*mat.shape)
        rand_arr = rand_arr < mutation_ratio
        mat = np.logical_xor(mat, rand_arr)

    def evaluation(self, k_lst):  # todo edit
        evaluation_lst = np.zeros(k_lst.shape[0], dtype=np.float_)
        for k_idx in range(k_lst.shape[0]):
            k = k_lst[k_idx, :, :]
            evaluation_lst[k_idx] = self.simulator.evaluate(k, self.pop_lst, 0.5)
            # msvc_idx = 0
            # evaluation_lst[k_idx] = 0.0
            # for svc_idx in range(self.svc_set.num_service):
            #     qos = self.pop_lst[svc_idx] / self.svc_set.svc_lst[svc_idx].deadline / self.svc_set.svc_lst[svc_idx].num_micro_service
            #     for i in range(self.svc_set.svc_lst[svc_idx].num_micro_service):
            #         if np.any(k[:, msvc_idx + i]):
            #             evaluation_lst[k_idx] += qos
            #     msvc_idx += self.svc_set.svc_lst[svc_idx].num_micro_service

        return evaluation_lst


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()  # hidden layer = 10
        laysers = [nn.Linear(input_size, 2**10), nn.ReLU()]
        for _ in range(4):
            laysers += [nn.Linear(2 ** 10, 2 ** 10), nn.ReLU()]
        laysers.append(nn.Linear(2 ** 10, output_size))
        self.layers = nn.Sequential(*laysers)
        # self.layer_1 = nn.Linear(input_size, 2 ** 10)
        # self.layer_2 = nn.Linear(2 ** 10, 2 ** 10)
        # self.layer_3 = nn.Linear(2 ** 10, 2 ** 10)
        # self.layer_4 = nn.Linear(2 ** 10, 2 ** 10)
        # self.layer_out = nn.Linear(2 ** 10, output_size)

    def forward(self, x):
        # x = self.layer_1(x)
        # x = nn.functional.relu(self.layer_2(x))
        # x = nn.functional.relu(self.layer_3(x))
        # x = nn.functional.relu(self.layer_4(x))
        # return self.layer_out(x)
        return self.layers(x)


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
        self.position = (self.position + 1) % self.capacity

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
TARGET_UPDATE = 100


class RL:
    def __init__(self, num_node, num_svc, num_msvc, state_manager, action_manager=None):
        self.state_manager = state_manager
        self.action_manager = action_manager
        self.model = None
        self.num_node = num_node
        self.num_svc = num_svc
        self.num_msvc = num_msvc
        # self.input_size = num_node * 2 + num_msvc * 2 + num_msvc
        self.input_size = state_manager.get_input_size()
        self.output_size = state_manager.get_output_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(input_size=self.input_size, output_size=self.output_size).to(self.device)
        self.target_net = DQN(input_size=self.input_size, output_size=self.output_size).to(self.device)
        self.output_indices = torch.arange(self.output_size, device=self.device).view(1, -1)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-6)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        # self.episode_reward = list()
        self.episode_reward = deque(maxlen=100)
        self.episode_loss = list()
        self.output_file = None

        self.svc_map_mat = np.ascontiguousarray(self.state_manager.svc_map_mat.reshape(1, self.num_svc, self.num_msvc), dtype=np.bool_)
        self.svc_map_mat = torch.from_numpy(self.svc_map_mat)
        self.svc_map_mat = self.svc_map_mat.to(self.device)

    def save_model(self, folder_path):
        torch.save(self.policy_net.state_dict(), os.path.join(folder_path, "policy_net"))
        torch.save(self.target_net.state_dict(), os.path.join(folder_path, "target_net"))

    def load_model(self, folder_path):
        self.policy_net.load_state_dict(torch.load(os.path.join(folder_path, "policy_net")))
        self.target_net.load_state_dict(torch.load(os.path.join(folder_path, "target_net")))

    def select_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1

        if sample > eps_threshold:
            with torch.no_grad():
                result = self.policy_net(state)

        else:
            result = torch.rand((1, self.output_size), device=self.device)
        return self.action_translate(result, state)

    def get_mask(self, state):
        start = 0
        end = 0
        end += self.num_node
        cpu_resource = state[:, start:end]
        start = end
        end += self.num_msvc
        cpu_require = state[:, start:end]
        start = end
        end += self.num_node
        mem_resource = state[:, start:end]
        start = end
        end += self.num_msvc
        mem_require = state[:, start:end]
        start = end
        end += self.num_msvc * self.num_node
        mat_k = state[:, start:end]
        mat_k = mat_k.type(torch.bool)
        mat_k = mat_k.view(-1, self.num_node, self.num_msvc)
        in_fog = mat_k.any(dim=1, keepdim=True)

        mask = cpu_resource.view(state.shape[0], -1, 1).ge(cpu_require.view(state.shape[0], 1, -1))
        mask = torch.logical_and(mask,
                                 mem_resource.view(state.shape[0], -1, 1).ge(mem_require.view(state.shape[0], 1, -1)))

        msvc_map = torch.logical_and(torch.logical_not(in_fog), self.svc_map_mat)
        # first_idx = (self.svc_map_mat == False).cumprod(dim=2).sim(dim=2)
        # first_idx[first_idx > self.num_msvc] = 0
        target_idx = torch.logical_not(msvc_map).cumprod(dim=2).sum(dim=2)
        target_idx = target_idx.unsqueeze(1)
        target_idx = target_idx.expand(-1, mask.shape[1], -1)
        # target_idx[target_idx > self.num_msvc] = first_idx[target_idx > self.num_msvc]
        msvc_mask = torch.zeros(mask.shape[0], mask.shape[1], mask.shape[2]+1, dtype=torch.bool, device=self.device)
        msvc_mask.scatter_(2, target_idx, True)
        msvc_mask = msvc_mask[:, :, :-1]
        # for i in range(mask.shape[0]):
        #     msvc_mask[i, :, target_idx[i, target_idx[i] < self.num_msvc]] = True
        mask = torch.logical_and(mask, msvc_mask)
        return mask

    def action_translate(self, action, state):
        mask = self.get_mask(state)
        mask = mask.view(action.shape[0], -1)
        return (action * mask).max(1)[1].view(1, 1)

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
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # state_action_values = self.policy_net(state_batch) * action_batch
        # state_action_values = state_action_values.sum(axis=1, keepdims=True)
        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
        # max(1)[0]으로 최고의 보상을 선택하십시오.
        # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()  # normal dqn
        # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_mask = self.get_mask(non_final_next_states)
        next_state_mask = next_state_mask.view(non_final_next_states.shape[0], -1)
        argmax_q = (self.policy_net(non_final_next_states) * next_state_mask).max(1, keepdim=True)[1]   # apply double dqn
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, argmax_q).view(-1)

        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Huber 손실 계산
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def run(self, period=60*10., episode_period=60*60, num_episodes=100):
        result = list()
        self.state_manager.set_episode_period(episode_period)
        for i_episode in range(num_episodes):
            # 환경과 상태 초기화
            epi_reward = 0.0
            self.state_manager.state_reset(rl=True)
            state = self.state_manager.get_state()

            state = state.reshape((1, state.size))
            state = np.ascontiguousarray(state, dtype=np.float32)  # nn.linear use float32
            state = torch.from_numpy(state)
            state = state.to(self.device)

            for t in count():
                # 행동 선택과 수행
                action = self.select_action(state)
                if self.action_manager is not None:
                    self.action_manager.set_action(action.cpu().data)  # set action
                self.state_manager.set_action(action.cpu().data)  # set action

                reward, done = self.state_manager.step(period, rl=True)
                epi_reward += reward
                next_state = self.state_manager.get_state()
                next_state = next_state.reshape((1, next_state.size))
                next_state = np.ascontiguousarray(next_state, dtype=np.float32) # nn.linear use float32
                next_state = torch.from_numpy(next_state)
                next_state = next_state.to(self.device)
                # 다음 상태로 이동
                state = next_state
                if done:
                    break
            result.append(epi_reward)

        print('Complete')
        return result

    def plot_durations(self):
        plt.figure(2)
        plt.clf()
        plt.title('Training...')
        plt.subplot(1, 2, 1)
        plt.xlabel('Episode')
        plt.ylabel('reward')
        reward_t = torch.tensor(self.episode_reward)
        plt.plot(reward_t.numpy())
        if len(reward_t) >= 100:
            means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.subplot(1, 2, 2)
        plt.xlabel('Episode')
        plt.ylabel('loss')
        loss_t = torch.tensor(self.episode_loss)
        plt.plot(loss_t.numpy())
        if len(loss_t) >= 100:
            means = loss_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())
        plt.pause(0.001)

    def train(self, period=60*10., episode_period=60*60):
        # plt.ion()
        self.output_file = open("output.txt", 'w')
        num_episodes = int(1e7)
        # reward = None
        best_reward = None

        self.state_manager.set_episode_period(episode_period)
        # self.svc_map_mat = np.ascontiguousarray(self.state_manager.svc_map_mat, dtype=np.bool_)
        # self.svc_map_mat = torch.from_numpy(self.svc_map_mat)
        # self.svc_map_mat = self.svc_map_mat.to(self.device)

        for i_episode in range(num_episodes):
            # 환경과 상태 초기화
            epi_reward = 0.
            self.state_manager.req_gen_reset()
            while True:
                self.state_manager.state_reset()
                arr_request, r_done = self.state_manager.req_step(period)
                state = self.state_manager.get_state()

                # state = state.reshape((1, 1, state.size))
                state = state.reshape((1, state.size))
                state = np.ascontiguousarray(state, dtype=np.float32)  # nn.linear use float32
                state = torch.from_numpy(state)
                state = state.to(self.device)

                while True:
                    # 행동 선택과 수행
                    action = self.select_action(state)
                    if self.action_manager is not None:
                        self.action_manager.set_action(action.cpu().data)  # set action
                    constraint_chk, changed = self.state_manager.set_action(action.cpu().data)  # set action

                    # last_id = self.db_conn.get_last_id()
                    reward, done = self.state_manager.step(arr_request, constraint_chk, changed)
                    epi_reward += reward
                    reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                    # reward = torch.tensor([reward], device=self.device, dtype=torch.float32)

                    # 새로운 상태 관찰
                    next_state = self.state_manager.get_state()
                    next_state = next_state.reshape((1, next_state.size))
                    next_state = np.ascontiguousarray(next_state, dtype=np.float32) # nn.linear use float32
                    next_state = torch.from_numpy(next_state)
                    next_state = next_state.to(self.device)

                    # 메모리에 변이 저장
                    self.memory.push(state, action, next_state, reward)

                    # 다음 상태로 이동
                    state = next_state

                    # 최적화 한단계 수행(목표 네트워크에서)
                    loss = self.optimize_model()

                    if done:
                        break

                if r_done:
                    # self.episode_reward.append(epi_reward)
                    self.episode_reward.append(epi_reward)
                    # num_record = len(self.episode_reward)
                    # if num_record > 1000:
                    #     self.episode_reward = self.episode_loss[-900:]   # to 900
                    break

            if i_episode % TARGET_UPDATE == TARGET_UPDATE - 1:
                loss = np.array(loss.cpu().data)
                # self.plot_durations()
                # folder = "save_epi_%s" % i_episode
                folder = "save_last_epi"
                folder = os.path.join(SAVE_FOLDER, folder)
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                self.target_net.load_state_dict(self.policy_net.state_dict())
                self.save_model(folder)
                mean_reward = np.mean(self.episode_reward)
                if best_reward is None or mean_reward > best_reward:
                    best_reward = mean_reward
                    self.save_model(BEST_SAVE)
                if epi_reward is not None:
                    print("episode %s \t reward = %s \t  mean rewards = %s \t loss = %s" % (i_episode, epi_reward, mean_reward, loss.mean()))
                    self.output_file.write("episode %s \t reward = %s \t  mean rewards = %s \t loss = %s \n" % (i_episode, epi_reward, mean_reward, loss.mean()))

        print('Complete')
