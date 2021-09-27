import pymysql
import numpy as np
from datetime import datetime, timedelta
import time

USER_ID = 'test'
USER_PW = 'test_pw'
DB_IP = 'localhost'
DB_NAME = 'test'
DB_PORT = 3306


class MyConnector:
    def __init__(self, ip=DB_IP, id=USER_ID, pw=USER_PW, name=DB_NAME, port=DB_PORT):
        self.conn = pymysql.connect(host=ip, user=id, password=pw, db=name, port=port)
        self.cursor = self.conn.cursor()
        self.episode_timer = None
        self.resource_map = None
        self.require_map = None
        self.mat_k = None
        self.num_node = None
        self.num_msvc = None
        self.num_svc = None
        self.svc_map = None
        self.svc_info = None
        self.episode_period = None
        self.timer = None

    def get_data(self, sql):
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        result = dict()
        for row in rows:
            result[row[0]] = row[1:]
        return result

    def get_svc_data(self):
        self.svc_info = self.get_data("SELECT * FROM svc_info")
        return self.svc_info

    def get_msvc_data(self):
        msvc_data = self.get_data("SELECT * FROM msvc_info")
        self.num_msvc = len(msvc_data)
        self.require_map = np.zeros((2, len(msvc_data)))
        self.svc_map = dict()

        for i in msvc_data:
            self.require_map[0, i - 1] = msvc_data[i][3]
            self.require_map[1, i - 1] = msvc_data[i][4]
            if msvc_data[i][0] not in self.svc_map:
                self.svc_map[msvc_data[i][0]] = list()
            self.svc_map[msvc_data[i][0]].append(i)

        self.num_svc = len(self.svc_map)

        for key in self.svc_map:
            self.svc_map[key] = np.array(self.svc_map[key])

        return msvc_data, self.require_map, self.svc_map

    def get_node_info(self):
        node_info = self.get_data("SELECT * FROM node_info")
        self.num_node = len(node_info)
        self.resource_map = np.zeros((self.num_node, 2))
        for i in node_info:
            self.resource_map[i - 1, 0] = node_info[i][1]
            self.resource_map[i - 1, 1] = node_info[i][2]
        return node_info, self.resource_map

    def request_start(self, svc_id):
        sql = "INSERT INTO request_log (svc_id, request_time, is_serviced) VALUES (%s, now(6), False)" % svc_id
        self.cursor.execute(sql)
        self.cursor.execute("SELECT LAST_INSERT_ID()")
        last_id = self.cursor.fetchall()
        self.conn.commit()
        return last_id[0]

    def request_end(self, request_id):
        self.cursor.execute(
            "UPDATE request_log SET service_time = now(6), is_serviced = True WHERE ID = %s" % request_id)
        self.conn.commit()
        return 0

    def get_request_logs(self, start, end=None):
        sql = "SELECT * FROM request_log where request_time >= %s" % start
        if end is not None:
            sql = sql + " AND request_time < %s" % end
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        return list(rows)

    def close(self):
        self.conn.close()

    def set_k(self, k):
        k = np.reshape(k, (self.num_node, self.num_msvc))
        self.mat_k = k

    def get_state(self, is_first=True):
        result = np.zeros(self.num_node * 2 + self.num_msvc * 2 + self.num_msvc, dtype=np.float_)
        # if self.episode_timer is None or is_first:
        #     return result
        temp_map = np.zeros(self.num_node + self.num_msvc, dtype=np.float_)  # for normalize
        temp_map[:self.num_node] = self.resource_map.T[0, :]  # cpu resource
        temp_map[self.num_node:] = self.require_map[0, :]  # cpu requirement
        result[:self.num_node + self.num_msvc] = temp_map / temp_map.max()  # cpu constraint

        temp_map[:self.num_node] = self.resource_map.T[1, :]  # mem resource
        temp_map[self.num_node:] = self.require_map[1, :]  # cpu requirement
        result[self.num_node + self.num_msvc:(self.num_node + self.num_msvc) * 2] = temp_map / temp_map.max()        # cpu constraint

        temp_map = np.zeros(self.num_msvc, dtype=np.float_)
        # if self.episode_timer is None or is_first:
        #     self.cursor.execute("SELECT * FROM request_log ORDER BY id desc LIMIT 100")
        # else:
        self.conn.commit()   # for sync
        self.cursor.execute("SELECT * FROM request_log where request_time > \"%s\"" % self.episode_timer)
        rows = self.cursor.fetchall()
        for row in rows:
            svc_idx = row[1]
            temp_map[(self.svc_map[svc_idx]-1)] = temp_map[(self.svc_map[svc_idx]-1)] + 1

        result[(self.num_node + self.num_msvc)*2:] = temp_map
        return result

    def set_episode_period(self, episode_period=3600):
        self.episode_period = episode_period
        if type(self.episode_period) in (int, float):
            self.episode_period = timedelta(seconds=self.episode_period)

    def state_reset(self):
        self.episode_timer = datetime.now()

    def get_last_id(self):
        self.cursor.execute("SELECT id FROM request_log ORDER BY id DESC LIMIT 1")
        return self.cursor.fetchall()[0][0]

    def step(self, period, alpha=2.0):
        time.sleep(period)
        sql = "SELECT * FROM request_log WHERE request_time > " \
              "DATE_SUB(now(6), INTERVAL %s SECOND) AND is_serviced = true" % period
        self.conn.commit()   # sync (because sleep)
        self.cursor.execute(sql)
        rows = self.cursor.fetchall()
        result = 0.0
        for row in rows:
            svc_idx = row[1]
            fmt_str = '%Y-%m-d %H:%M:%S.%f'
            time_delta = row[3] - row[2]
            dead_line = timedelta(seconds=self.svc_info[svc_idx][1])
            if time_delta <= dead_line:
                result += 1.0
            elif time_delta <= alpha * dead_line:
                result += ((alpha * dead_line.total_seconds() - time_delta.total_seconds()) / (alpha - 1.0))
            else:
                result += 0.0

        return result, datetime.now() - self.episode_timer >= self.episode_period
