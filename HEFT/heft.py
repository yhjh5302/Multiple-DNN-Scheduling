import numpy as np

class HEFT:
    def __init__(self, dataset):
        self.system_manager = dataset.system_manager
        self.num_servers = dataset.num_servers
        self.num_services = dataset.num_services
        self.num_partitions = dataset.num_partitions

    def run_algo(self, timeslot=0):
        x = np.full(shape=self.num_partitions, fill_value=self.system_manager.cloud_id, dtype=np.int32)
        y = np.arange(start=0, stop=self.num_partitions, dtype=np.int32)
        server_lst = list(self.system_manager.edge.keys()) + list(self.system_manager.fog.keys())
        scheduling_lst = np.array(sorted(zip(self.system_manager.ranku, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1]

        for top_rank in scheduling_lst:
            # initialize the earliest finish time of the task
            earliest_finish_time = np.inf
            # for all available server, find earliest finish time server
            for s_id in server_lst:
                temp_x = x[top_rank]
                x[top_rank] = s_id
                if self.system_manager.constraint_chk(deployed_server=x, s_id=s_id):
                    self.system_manager.set_env(deployed_server=x, execution_order=scheduling_lst)
                    finish_time = self.system_manager.get_completion_time(top_rank)
                    if finish_time < earliest_finish_time:
                        earliest_finish_time = finish_time
                    else:
                        x[top_rank] = temp_x
                else:
                    x[top_rank] = temp_x
        return np.array(x, dtype=np.int32), np.array(y, dtype=np.int32)