import requests
import database
import threading
import random
import time


DB_IP = "165.132.107.7"
DB_ID = "test"
DB_PW = "test_pw"
DB_NAME = 'test'
REQUEST_RATE = 1.0


def key_thread_func(a_list):
    input("press enter to stop:\n")
    a_list.append(True)


def request_func(http_addr, svc_id):
    db_conn = database.MyConnector(DB_IP, DB_ID, DB_PW, DB_NAME)
    req_id = db_conn.request_start(svc_id)
    r = requests.get(http_addr)
    if r.text == 'end':
        db_conn.request_end(req_id)
    db_conn.close()


if __name__ == "__main__":
    db_conn = database.MyConnector(DB_IP, DB_ID, DB_PW, DB_NAME)
    msvc_info, _, svc_map = db_conn.get_msvc_data()
    db_conn.close()
    cum_weight = [1/i for i in range(1, len(svc_map)+1)]
    svc_lst = list(svc_map.keys())
    for i in range(len(cum_weight) - 1):
        cum_weight[i+1] += cum_weight[i]

    while True:
        request_rate = input("input request_rate (per minute):")
        try:
            request_rate = float(request_rate)
        except ValueError:
            request_rate = REQUEST_RATE

        a_list = list()
        t = threading.Thread(target=key_thread_func, args=[a_list])
        t.daemon = True  # daemon thread http://pythonstudy.xyz/python/article/24-%EC%93%B0%EB%A0%88%EB%93%9C-Thread
        t.start()
        while not a_list:
            svc_key = random.choices(svc_lst, cum_weights=cum_weight)[0]
            addr = "http://test-svc-" + str(svc_key) + "-msvc-" + str(svc_map[svc_key][0]) + ":4000"
            # print(addr)
            # addr = "http://localhost:8888"
            t_r = threading.Thread(target=request_func, args=[addr, svc_key])
            t_r.daemon = True
            t_r.start()
            time.sleep(60. / request_rate)  # per minute

