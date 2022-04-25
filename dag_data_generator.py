from copy import deepcopy
import numpy as np
import random
from dag_server import *
from multilevel_graph_partitioning import MultiLevelGraphPartitioning


class DAGDataSet:
    def __init__(self, num_timeslots):
        self.num_timeslots = num_timeslots
        self.apply_partition = True
        self.svc_set, self.system_manager = self.data_gen()
        self.coarsened_graph, self.tran_time, self.proc_time = self.graph_coarsening(num_partitions=50)
        
        # test
        # y = np.array(sorted(zip(self.system_manager.rank_u, np.arange(self.num_partitions)), reverse=True), dtype=np.int32)[:,1]
        # x = np.random.randint(low=self.num_servers-2, high=self.num_servers-1, size=(self.num_partitions))
        # self.system_manager.set_env(deployed_server=x, execution_order=y)
        # print('total_time', self.system_manager.total_time_dp())
        # print('tran_time', sum(self.tran_time.values()))
        # print('proc_time', sum(self.proc_time))
        # input()

    def create_arrival_rate(self, num_services, minimum, maximum):
        return minimum + (maximum - minimum) * np.random.random(num_services)

    def cnn_partitioning(self, layer_info, min_unit):
        if layer_info['layer_name'] == 'conv1':
            min_unit = 4
        elif layer_info['layer_name'] == 'maxpool1' or layer_info['layer_name'] == 'conv2' or layer_info['layer_name'] == 'conv3':
            min_unit = 2

        min_unit_partitions = []
        output_data_location = []
        num_partitions = math.floor(layer_info['output_height'] / min_unit)
        for idx in range(num_partitions):
            partition = deepcopy(layer_info)
            partition['layer_name'] = layer_info['layer_name'] + '_{:d}'.format(idx)
            partition['layer_idx'] = idx
            partition['original_layer_name'] = layer_info['layer_name']
            partition['output_height'] = min_unit if idx < (num_partitions - 1) else (min_unit + (layer_info['output_height'] % min_unit))
            partition['input_height'] = (partition['output_height'] - 1) * layer_info['stride'] + partition['kernel'] - max(layer_info['padding'] - layer_info['stride'] * min_unit * idx, (layer_info['padding'] - (layer_info['input_height'] + 1) % layer_info['stride']) - layer_info['stride'] * min_unit * (num_partitions - 1 - idx), 0)
            start = max(layer_info['stride'] * min_unit * idx - layer_info['padding'], 0)
            end = start + partition['input_height']
            partition['input_data_location'] = [i for i in range(start, end)]
            partition['input_data_size'] = partition['input_height'] * partition['input_width'] * partition['input_channel'] * 4
            partition['workload_size'] = layer_info['workload_size'] * (partition['output_height'] / layer_info['output_height'])
            output_data_location += [partition['layer_name'] for _ in range(partition['output_height'])]
            min_unit_partitions.append(partition)
        return min_unit_partitions, output_data_location

    def fc_partitioning(self, layer_info, min_unit):
        min_unit_partitions = []
        num_partitions = math.floor(layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] / min_unit)
        for idx in range(num_partitions):
            partition = deepcopy(layer_info)
            partition['layer_name'] = layer_info['layer_name'] + '_{:d}'.format(idx)
            partition['layer_idx'] = idx
            partition['original_layer_name'] = layer_info['layer_name']
            partition['input_height'] = 1
            partition['input_width'] = 1
            partition['input_channel'] = min_unit
            partition['workload_size'] = layer_info['workload_size'] / num_partitions
            min_unit_partitions.append(partition)
        return min_unit_partitions

    def data_gen(self):
        import config
        system_manager = SystemManager()

        # create service set
        svc_set = ServiceSet()
        for dnn in config.service_info:
            svc = Service(dnn['deadline'])

            # just partitioning layers into minimum unit partitions
            if self.apply_partition:
                partitioned_layers = []
                for layer_info in dnn['layers']:
                    if layer_info['layer_type'] == 'cnn' or layer_info['layer_type'] == 'maxpool':
                        min_unit_partitions, output_data_location = self.cnn_partitioning(layer_info, min_unit=1)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'min_unit_partitions':min_unit_partitions, 'output_data_location':output_data_location})
                    elif layer_info['layer_type'] == 'fc' or layer_info['layer_type'] == 'activation':
                        min_unit_partitions = self.fc_partitioning(layer_info, min_unit=768)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'min_unit_partitions':min_unit_partitions, 'predecessors':layer_info['predecessors']})
                    else:
                        partitioned_layers.append(layer_info)
                # predecessor recalculation for the minimum unit partitions
                partitions = []
                for layer_info in partitioned_layers:
                    if layer_info['layer_type'] == 'cnn' or layer_info['layer_type'] == 'maxpool':
                        for partition in layer_info['min_unit_partitions']:
                            predecessors = []
                            input_data_size = []
                            for pred_layer_name in partition['predecessors']:
                                pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                                for input_location in partition['input_data_location']:
                                    # find predecessor partition and calculate input size from the predecessor
                                    pred_partition = next(l for l in pred_layer['min_unit_partitions'] if l['layer_name'] == pred_layer['output_data_location'][input_location])
                                    if pred_partition['layer_name'] not in predecessors:
                                        predecessors.append(pred_partition['layer_name'])
                                        input_data_size.append(partition['input_width'] * partition['input_channel'] * 4)
                                    else:
                                        input_data_size[predecessors.index(pred_partition['layer_name'])] += partition['input_width'] * partition['input_channel'] * 4
                            partition['predecessors'] = predecessors
                            if len(predecessors) > 0 :
                                partition['input_data_size'] = input_data_size
                            partition['successors'] = []
                            partition['output_data_size'] = []
                            del partition['input_data_location']
                            partitions.append(partition)
                    elif layer_info['layer_type'] == 'fc':
                        predecessors = []
                        for pred_layer_name in layer_info['predecessors']:
                            pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                            for pred_partition in pred_layer['min_unit_partitions']:
                                predecessors.append((pred_partition['layer_name'], pred_partition['output_height'] * pred_partition['output_width'] * pred_partition['output_channel']))
                        idx = 0
                        for (name, output) in predecessors:
                            while output:
                                partition = layer_info['min_unit_partitions'][idx]
                                partition['predecessors'] = [name]
                                partition['input_data_size'] = [partition['input_channel'] * 4]
                                partitions.append(partition)
                                if output % partition['input_channel'] > 0:
                                    raise RuntimeError("minimum unit error!")
                                output -= partition['input_channel']
                                idx += 1
                    elif layer_info['layer_type'] == 'activation':
                        predecessors = []
                        for pred_layer_name in layer_info['predecessors']:
                            pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                            for pred_partition in pred_layer['min_unit_partitions']:
                                predecessors.append(pred_partition['layer_name'])
                        for partition in layer_info['min_unit_partitions']:
                            partition['predecessors'] = []
                            partition['input_data_size'] = []
                            for name in predecessors:
                                partition['predecessors'].append(name)
                                partition['input_data_size'].append(partition['input_channel'] * 4)
                            partitions.append(partition)
                    else:
                        partitions.append(layer_info)
                # successor recalculation for the minimum unit partitions
                for partition in partitions:
                    for ith, pred_partition_name in enumerate(partition['predecessors']):
                        pred_partition = next(p for p in partitions if p['layer_name'] == pred_partition_name)
                        pred_partition['successors'].append(partition['layer_name'])
                        pred_partition['output_data_size'].append(partition['input_data_size'][ith])
                # create partitions
                for partition in partitions:
                    if len(partition['predecessors']) == 0 and len(partition['successors']) == 0:
                        print(partition['layer_name'], 'has no predecessor and successor node. so deleted from DAG')
                        continue
                    svc.partitions.append(Partition(service=svc, **partition))
            else:
                for layer_info in dnn['layers']:
                    svc.partitions.append(Partition(service=svc, **layer_info))

            svc_set.add_services(svc)

            # predecessor, successor check
            for p_id, partition in enumerate(svc.partitions):
                for i in range(len(partition.predecessors)):
                    for c in svc.partitions:
                        if c.layer_name == partition.predecessors[i]:
                            partition.predecessors[i] = c
                for i in range(len(partition.successors)):
                    for c in svc.partitions:
                        if c.layer_name == partition.successors[i]:
                            partition.successors[i] = c
                if p_id != partition.id:
                    raise RuntimeError("Partitions unordered! You have to sort the partition list in id order.")
                svc.partition_computation_amount.append(partition.workload_size)
                svc.partition_predecessor.append([pred.id for pred in partition.predecessors])
                svc.partition_successor.append([succ.id for succ in partition.successors])
            
            # predecessor, successor data size check
            for partition in svc.partitions:
                if len(partition.predecessors) > 0:
                    for i in range(len(partition.predecessors)):
                        svc.input_data_size[(partition.predecessors[i].id,partition.id)] = partition.input_data_size[i]
                else:
                    svc.input_data_size[(partition.id,partition.id)] = partition.input_data_size
                if len(partition.successors) > 0:
                    for i in range(len(partition.successors)):
                        svc.output_data_size[(partition.id,partition.successors[i].id)] = partition.output_data_size[i]
                else:
                    svc.output_data_size[(partition.id,partition.id)] = partition.output_data_size

            # dag structure error check
            for p1 in svc.partitions:
                for p2 in svc.partitions:
                    if ((p1.id,p2.id) in svc.input_data_size.keys() and (p2.id,p1.id) in svc.output_data_size.keys()) and svc.input_data_size[(p1.id,p2.id)] != svc.output_data_size[(p2.id,p1.id)]:
                        raise RuntimeError("DAG input output data mismatch!!", (p1.layer_name, p2.layer_name))

        self.num_services = len(svc_set.services)
        self.num_partitions = len(svc_set.partitions)

        # create arrival rate table
        self.max_arrival = 50
        self.min_arrival = 10

        svc_arrival = list()
        for t in range(self.num_timeslots):
            svc_arrival.append(self.create_arrival_rate(num_services=self.num_services, minimum=self.min_arrival, maximum=self.max_arrival))

        # create servers
        local = dict()
        edge = dict()
        cloud = dict()

        self.num_locals = len(config.local_device_info)
        self.num_edges = len(config.edge_server_info)
        self.num_clouds = len(config.cloud_server_info)
        self.num_servers = self.num_locals + self.num_edges + self.num_clouds
        id = 0
        for local_device in config.local_device_info:
            local[id] = Server(**local_device, system_manager=system_manager, id=id)
            id += 1
        for edge_server in config.edge_server_info:
            edge[id] = Server(**edge_server, system_manager=system_manager, id=id)
            id += 1
        for cloud_server in config.cloud_server_info:
            cloud[id] = Server(**cloud_server, system_manager=system_manager, id=id)
            id += 1

        # create network manager
        net_manager = NetworkManager(channel_bandwidth=1024*1024*40, channel_gain=1, gaussian_noise=1, B_edge=1024*1024*25, B_cloud=1024*1024*1, local=local, edge=edge, cloud=cloud)
        net_manager.P_dd = np.zeros(shape=(self.num_servers, self.num_servers))
        for i in range(self.num_servers):
            for j in range(i + 1, self.num_servers):
                net_manager.P_dd[i, j] = net_manager.P_dd[j, i] = random.uniform(0.5, 1)
            net_manager.P_dd[i, i] = 0
        net_manager.cal_b_dd()

        # init system manager
        system_manager.net_manager = net_manager
        system_manager.num_servers = self.num_servers
        system_manager.num_services = self.num_services
        system_manager.num_partitions = self.num_partitions
        system_manager.set_service_set(svc_set, svc_arrival, self.max_arrival)
        system_manager.set_servers(local, edge, cloud)

        system_manager.rank_u = np.zeros(self.num_partitions)
        system_manager.calc_average()
        for svc in svc_set.services:
            for partition in svc.partitions:
                system_manager.calc_rank_u(partition)

        return svc_set, system_manager

    def graph_coarsening(self, num_partitions):
        # dag coarsening and error checking
        for svc in self.svc_set.services:

            # graph partitioning algorithm
            p_algo = MultiLevelGraphPartitioning(dataset=self, num_partitions=num_partitions)
            return p_algo.run_algo()
            
            # predecessor, successor check
            # for local search optimization
            for partition in svc.partitions:
                if len(partition.predecessors) == 0:
                    partition.find_total_predecessors()

                if len(partition.successors) == 0:
                    partition.find_total_successors()

            # for local search optimization
            for p in svc.partitions:
                predcessors_id = [pred.id for pred in p.total_predecessors]
                successors_id = [succ.id for succ in p.total_successors]
                p.total_pred_succ_id = predcessors_id + successors_id