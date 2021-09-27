import os, shutil
import time
import yaml
import numpy as np
import subprocess
import re
from datetime import datetime, timedelta
from svc_algorithm import algorithm, sim_system
import database
from linux_script import linux_apply


DATABASE_ADDRESS = "165.132.107.7"  # todo change
SVC_YAML_FOLDER = "./svc_yaml"
SAVE_FOLDER = "model_save_3"
BEST_SAVE = "model_save_3/best"
TEST_SAVE = "test_data"
EPI_NUM = 10

if __name__ == "__main__":
    mat_k = None
    node_status = linux_apply.NodeStatus()
    db_conn = database.MyConnector(ip=DATABASE_ADDRESS)
    svc_info = db_conn.get_svc_data()

    r_info = [
        (1 * 3600., 60), (2 * 3600., 50), (3 * 3600., 36), (4 * 3600., 22), (5 * 3600., 18), (6 * 3600., 22),
        (7 * 3600., 40), (8 * 3600., 60), (9 * 3600., 70), (10 * 3600., 67), (11 * 3600., 66), (12 * 3600., 50),
        (13 * 3600., 58), (14 * 3600., 55), (15 * 3600., 65), (16 * 3600., 70), (17 * 3600., 78), (18 * 3600., 98),
        (19 * 3600., 110), (20 * 3600., 120), (21 * 3600., 115), (22 * 3600., 110), (23 * 3600., 80), (24 * 3600., 70)
    ]
    request_gen = sim_system.RequestGenerator2(r_info, 24.*60.*60., len(svc_info), 47, 5 * 60)
    request_gen.save_file(os.path.join(TEST_SAVE, "request.dump"))
    request_gen.set_using_record(True)
    request_gen.make_random_record(EPI_NUM, 60. * 60)
    state_manager = sim_system.SimpleSimulator(db_conn, request_gen)
    state_manager.state_reset()
    # k8s_manager = K8S_Manager(db_conn)
    # rl = algorithm.RL(db_conn.num_node, db_conn.num_svc, db_conn.num_msvc, db_conn, k8s_manager)
    # rl.train(period=60 * 10., episode_period=60*30.)
    rl = algorithm.RL(db_conn.num_node, db_conn.num_svc, db_conn.num_msvc, state_manager)
    rl.load_model(BEST_SAVE)
    rl_result = rl.run(period=60 * 60., episode_period=60*60*24, num_episodes=EPI_NUM)
    # rl.train(period=60 * 60., episode_period=60*60*24)
    ga_result = list()
    for epi_idx in range(EPI_NUM):
        state_manager.state_reset()
        # _, r_arr = request_gen.step(24 * 60 * 60)
        pop = request_gen.get_pop()
        ga = algorithm.Memetic(pop, state_manager)
        # mat_k = ga.run_algo(20, int(1e5))
        mat_k = ga.run_algo(100, int(1e2))
        print(state_manager.constraint_chk(mat_k))
        state_manager.set_k(mat_k)
        epi_reward = 0.0
        done = False
        state_manager.request_gen.record_epi_idx = 0
        while not done:
            reward, done = state_manager.step(60 * 60)
            epi_reward += reward
        ga_result.append(epi_reward)

    print("RL: %s \n GA: %s" % (np.mean(rl_result), np.mean(ga_result)))
    print(rl_result)
    print('---------------------------------------------------')
    print(ga_result)
