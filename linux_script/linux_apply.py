import os, shutil
import time
import yaml
import numpy as np
import subprocess
import re
from datetime import datetime, timedelta
from svc_algorithm import algorithm, sim_system
import database
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from config import *


class NodeStatus:
    def __init__(self):
        self.status = None
        self.last_chk_time = None
        self.out_r = re.compile("(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)")

    def update_node_info(self):
        try:
            out_str = subprocess.check_output(["kubectl", "get", "nodes"], universal_newlines=True)
            out_str = out_str.replace('\r\n', '\n')
            out_str = out_str.replace('\r', '\n')
        except subprocess.CalledProcessError as e:
            print("error occurred")
            e.stdout()
            out_str = None
        self.status = list()
        if out_str is not None:
            lines = out_str.split('\n')
            matchobj = self.out_r.search(lines[0])
            fields = list()
            for idx in range(1, matchobj.lastindex+1):
                fields.append(matchobj[idx])

            for line in lines[1:-1]:
                matchobj = self.out_r.search(line)
                temp_dict = dict()

                for idx in range(matchobj.lastindex):
                    temp_dict[fields[idx]] = matchobj[idx + 1]

                self.status.append(temp_dict)
        self.last_chk_time = datetime.now()

    def filtering(self, filtering_roles='master', negative_filter=True):  # negative = remove filter item,
        remove_lst = list()                                               # positive = remove unfiltered item
        for idx in range(len(self.status)):
            filtered = False
            if type(filtering_roles) is not list:
                filtering_roles = [filtering_roles]
            for filtering_role in filtering_roles:
                if self.status[idx]['ROLES'].find(filtering_role) > -1:
                    filtered = True
                    break

            if negative_filter == filtered:  # xor operation
                remove_lst.append(idx)

        for idx in reversed(remove_lst):
            del self.status[idx]


class K8S_Manager:
    def __init__(self, db_conn):
        self.db_conn = db_conn
        self.SVC_YAML_FOLDER = "./svc_yaml/"
        self.node_info = None
        self.resource_map = None
        self.msvc_info = None
        self.require_map = None
        self.svc_map = None
        self.mat_k = None
        self.constraint_chk = True
        self.yaml_r = re.compile("msvc-(\d+).yaml")

    def set_k(self, mat_k):
        self.node_info, self.resource_map = self.db_conn.get_node_info()
        self.msvc_info, self.require_map, self.svc_map = self.db_conn.get_msvc_data()
        mat_k = np.reshape(mat_k, (self.db_conn.num_node, self.db_conn.num_msvc))

        cpu_need = mat_k * self.require_map[0, :]
        cpu_need = np.sum(cpu_need, axis=1)

        mem_need = mat_k * self.require_map[1, :]
        mem_need = np.sum(mem_need, axis=1)

        self.constraint_chk = np.logical_and(cpu_need <= self.resource_map[:, 0], mem_need <= self.resource_map[:, 1])
        if self.constraint_chk.all():
            for svc_folder in os.listdir(SVC_YAML_FOLDER):
                svc_path = os.path.join(SVC_YAML_FOLDER, svc_folder)
                if os.path.isdir(os.path.join(SVC_YAML_FOLDER, svc_folder)):
                    for msvc_file in os.listdir(svc_path):
                        m = self.yaml_r.match(msvc_file)
                        if m is not None:
                            msvc_idx = int(m[1]) - 1
                            full_path = os.path.join(svc_path, msvc_file)

                            with open(full_path, 'r') as rf:
                                data = yaml.safe_load_all(rf)
                                data = list(data)
                            is_fog = np.any(mat_k[:, msvc_idx])
                            try:
                                fog_arg_idx = data[0]['spec']['template']['spec']['containers'][0]['args'].index('--is_fog')
                                data[0]['spec']['template']['spec']['containers'][0]['args'][fog_arg_idx + 1] = str(is_fog)
                            except AttributeError:
                                data[0]['spec']['template']['spec']['containers'][0]['args'] = \
                                    data[0]['spec']['template']['spec']['containers'][0]['args'] + ['--is_fog', str(is_fog)]

                            if is_fog:
                                data[0]['spec']['template']['spec']['nodeSelector'] = {
                                    'kubernetes.io/hostname': self.node_info[np.where(mat_k[:, msvc_idx])[0][0]+1][0]
                                }

                            else:
                                if 'nodeSelector' in data[0]['spec']['template']['spec']:
                                    del data[0]['spec']['template']['spec']['nodeSelector']

                            with open(full_path, 'w') as wf:
                                yaml.safe_dump_all(data, wf)

            # output = subprocess.check_output("pwd", cwd=os.getcwd())
            # print(output)
            # output = subprocess.check_output("ls -a .", cwd=os.getcwd(), shell=True)
            # print(output)
            output = subprocess.check_output("kubectl apply -f %s -R" % SVC_YAML_FOLDER, cwd=os.getcwd(), shell=True)
            print("kubernetes output:")
            print(output.decode("utf-8"))
            print("-----------------------------------------------------------------------------------------------")


def generate_random_request_rate(svc_info, request_rate):
    result = list()
    period = 24 * 60 * 60 / len(request_rate)
    zipf_dist = np.array([1/i for i in range(1, len(svc_info) + 1)])
    zipf_dist /= zipf_dist.sum()
    end_t = 0.
    for r in request_rate:
        end_t += period
        if r % 6 == 0:
            np.random.shuffle(zipf_dist)
        r_arr = r * zipf_dist
        result.append((end_t, r_arr))
    return result


if __name__ == "__main__":
    if os.path.isdir(SAVE_FOLDER):
        shutil.rmtree(SAVE_FOLDER)
    os.mkdir(SAVE_FOLDER)
    os.mkdir(BEST_SAVE)

    mat_k = None
    node_status = NodeStatus()
    db_conn = database.MyConnector(ip=DATABASE_ADDRESS)
    # db_conn.set_episode_period()
    # node_info = db_conn.get_node_info()
    # msvc_data, require_map, svc_map = db_conn.get_msvc_data()
    svc_info = db_conn.get_svc_data()

    # r_arr = [
    #     1/6, 1/6, 1/6, 1/6, 1/6, 1/5, 1/4, 1/4, 1/4, 1/4, 1/3, 1/3,
    #     1/3, 1/3, 1/3, 1/4, 1/4, 1/4, 1/4, 1/4, 1/5, 1/6, 1/6, 1/6
    #          ]
    # request_info = generate_random_request_rate(svc_info, r_arr)
    # request_gen = sim_system.RequestGenerator(request_info, 24*60*60)
    r_info = [
        (1 * 3600., 60), (2 * 3600., 50), (3 * 3600., 36), (4 * 3600., 22), (5 * 3600., 18), (6 * 3600., 22),
        (7 * 3600., 40), (8 * 3600., 60), (9 * 3600., 70), (10 * 3600., 67), (11 * 3600., 66), (12 * 3600., 50),
        (13 * 3600., 58), (14 * 3600., 55), (15 * 3600., 65), (16 * 3600., 70), (17 * 3600., 78), (18 * 3600., 98),
        (19 * 3600., 110), (20 * 3600., 120), (21 * 3600., 115), (22 * 3600., 110), (23 * 3600., 80), (24 * 3600., 70)
    ]
    request_gen = sim_system.RequestGenerator2(r_info, 24.*60.*60., len(svc_info), 47, 5 * 60)
    request_gen.save_file(os.path.join(SAVE_FOLDER, "request.dump"))
    state_manager = sim_system.SimpleSimulator(db_conn, request_gen)
    state_manager.state_reset()
    # k8s_manager = K8S_Manager(db_conn)
    # rl = algorithm.RL(db_conn.num_node, db_conn.num_svc, db_conn.num_msvc, db_conn, k8s_manager)
    # rl.train(period=60 * 10., episode_period=60*30.)
    rl = algorithm.RL(db_conn.num_node, db_conn.num_svc, db_conn.num_msvc, state_manager)
    rl.train(period=60 * 60., episode_period=60*60*24)



