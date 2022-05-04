from copy import deepcopy
import numpy as np
import random
import config
from dag_server import *
from multilevel_graph_partitioning import MultiLevelGraphPartitioning


class DAGDataSet:
    def __init__(self, num_timeslots=1, num_partitions=[32,32,20,10], apply_partition=True, net_manager=None, svc_arrival=None):
        self.num_timeslots = num_timeslots
        self.apply_partition = apply_partition
        self.svc_set, self.system_manager = self.data_gen(net_manager=net_manager, svc_arrival=svc_arrival)
        self.coarsened_graph, self.tran_time, self.proc_time = self.graph_coarsening(num_partitions=num_partitions)
        
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
            partition['workload_size'] *= partition['output_height'] / layer_info['output_height']
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
            partition['workload_size'] /= num_partitions
            min_unit_partitions.append(partition)
        return min_unit_partitions

    def data_gen(self, net_manager=None, svc_arrival=None):
        self.service_info = config.service_info
        self.application = 0 # 0: GoogLeNet, 1: AlexNet, 2: ResNet-50, 3: Bert-3
        self.service_info = [deepcopy(self.service_info[self.application])]
        system_manager = SystemManager()

        # create service set
        svc_set = ServiceSet()
        for dnn in self.service_info:
            svc = Service(dnn['deadline'])
            
            # workload size calculation
            for layer_info in dnn['layers']:
                if layer_info['layer_type'] == 'cnn':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * layer_info['input_channel'] * layer_info['kernel'] * layer_info['kernel']
                elif layer_info['layer_type'] == 'maxpool' or layer_info['layer_type'] == 'avgpool':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * layer_info['kernel'] * layer_info['kernel']
                elif layer_info['layer_type'] == 'fc':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * (2 * layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] - 1)

            # just partitioning layers into minimum unit partitions
            if self.apply_partition:
                partitioned_layers = []
                for layer_info in dnn['layers']:
                    if layer_info['layer_type'] == 'cnn' or layer_info['layer_type'] == 'maxpool' or layer_info['layer_type'] == 'avgpool':
                        if dnn['model_name'] == 'GoogLeNet':
                            min_unit = max(math.floor(layer_info['output_height'] / 27), 1)
                        elif dnn['model_name'] == 'AlexNet':
                            min_unit = max(math.floor(layer_info['output_height'] / 27), 1)
                        elif dnn['model_name'] == 'ResNet-50':
                            min_unit = max(math.floor(layer_info['output_height'] / 14), 1)
                        else:
                            min_unit = 1

                        min_unit_partitions, output_data_location = self.cnn_partitioning(layer_info, min_unit=min_unit)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'min_unit_partitions':min_unit_partitions, 'output_data_location':output_data_location})
                    elif layer_info['layer_type'] == 'fc' and layer_info['is_first_fc'] == True:
                        if dnn['model_name'] == 'GoogLeNet':
                            min_unit = 768
                        elif dnn['model_name'] == 'AlexNet':
                            min_unit = 1536
                        elif dnn['model_name'] == 'ResNet-50':
                            min_unit = 2048
                        else:
                            min_unit = 256
                        min_unit_partitions = self.fc_partitioning(layer_info, min_unit=min_unit)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'is_first_fc':layer_info['is_first_fc'], 'min_unit_partitions':min_unit_partitions, 'predecessors':layer_info['predecessors']})
                    elif layer_info['layer_type'] == 'fc' and layer_info['is_first_fc'] == False:
                        if dnn['model_name'] == 'AlexNet':
                            min_unit = 682
                        else:
                            min_unit = 256
                        min_unit_partitions = self.fc_partitioning(layer_info, min_unit=min_unit)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'is_first_fc':layer_info['is_first_fc'], 'min_unit_partitions':min_unit_partitions, 'predecessors':layer_info['predecessors']})
                    else:
                        partitioned_layers.append(layer_info)
                # predecessor recalculation for the minimum unit partitions
                partitions = []
                for layer_info in partitioned_layers:
                    if layer_info['layer_type'] == 'cnn' or layer_info['layer_type'] == 'maxpool' or layer_info['layer_type'] == 'avgpool':
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
                    elif layer_info['layer_type'] == 'fc' and layer_info['is_first_fc'] == True:
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
                                partition['successors'] = []
                                partition['output_data_size'] = []
                                partitions.append(partition)
                                if output % partition['input_channel'] > 0:
                                    raise RuntimeError("minimum unit error!")
                                output -= partition['input_channel']
                                idx += 1
                    elif layer_info['layer_type'] == 'fc' and layer_info['is_first_fc'] == False:
                        predecessors = []
                        for pred_layer_name in layer_info['predecessors']:
                            pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                            for pred_partition in pred_layer['min_unit_partitions']:
                                predecessors.append((pred_partition['layer_name'], pred_partition['output_height'] * pred_partition['output_width'] * pred_partition['output_channel']))
                        for partition in layer_info['min_unit_partitions']:
                            partition['predecessors'] = list()
                            partition['input_data_size'] = list()
                            for (name, output) in predecessors:
                                partition['predecessors'].append(name)
                                partition['input_data_size'].append(output * 4)
                            partition['successors'] = []
                            partition['output_data_size'] = []
                            partitions.append(partition)
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
            for partition in svc.partitions:
                for i in range(len(partition.predecessors)):
                    for c in svc.partitions:
                        if c.layer_name == partition.predecessors[i]:
                            partition.predecessors[i] = c
                for i in range(len(partition.successors)):
                    for c in svc.partitions:
                        if c.layer_name == partition.successors[i]:
                            partition.successors[i] = c
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
        self.max_arrival = 10
        self.min_arrival = 10

        if svc_arrival == None:
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
            local_device = deepcopy(local_device)
            local_device['computing_intensity'] = local_device['computing_intensity'][self.application]
            local[id] = Server(**local_device, system_manager=system_manager, id=id)
            id += 1
        for edge_server in config.edge_server_info:
            edge_server = deepcopy(edge_server)
            edge_server['computing_intensity'] = edge_server['computing_intensity'][self.application]
            edge[id] = Server(**edge_server, system_manager=system_manager, id=id)
            id += 1
        for cloud_server in config.cloud_server_info:
            cloud_server = deepcopy(cloud_server)
            cloud_server['computing_intensity'] = cloud_server['computing_intensity'][self.application]
            cloud[id] = Server(**cloud_server, system_manager=system_manager, id=id)
            id += 1

        # create network manager
        if net_manager == None:
            net_manager = NetworkManager(channel_bandwidth=1024*1024*15, channel_gain=1, gaussian_noise=1, B_edge=1024*1024*3, B_cloud=1024*1024*1, local=local, edge=edge, cloud=cloud, request_device=local[0].id)
            net_manager.P_dd = np.zeros(shape=(self.num_servers, self.num_servers))
            for i in range(self.num_servers):
                for j in range(i + 1, self.num_servers):
                    net_manager.P_dd[i, j] = net_manager.P_dd[j, i] = random.uniform(0.666, 1)
                net_manager.P_dd[i, i] = 0
            net_manager.cal_b_dd()

        # init system manager
        system_manager.net_manager = net_manager
        system_manager.num_servers = self.num_servers
        system_manager.num_services = self.num_services
        system_manager.num_partitions = self.num_partitions
        system_manager.set_service_set(svc_set, svc_arrival)
        system_manager.set_servers(local, edge, cloud)

        system_manager.rank_u = np.zeros(self.num_partitions)
        system_manager.calc_average()
        for svc in svc_set.services:
            for partition in svc.partitions:
                system_manager.calc_rank_u(partition)

        return svc_set, system_manager

    def graph_coarsening(self, num_partitions):
        # dag coarsening and error checking
        coarsened_graph = []
        tran_time = []
        proc_time = []
        for svc in self.svc_set.services:

            # graph partitioning algorithm
            p_algo = MultiLevelGraphPartitioning(dataset=self)
            cg, T_tran, T_proc = p_algo.run_algo(svc=svc, num_partitions=num_partitions[self.application])
            coarsened_graph.append(cg)
            tran_time.append(T_tran)
            proc_time.append(T_proc)
        
        return coarsened_graph[0], tran_time[0], proc_time[0]