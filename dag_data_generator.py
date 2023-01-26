from copy import deepcopy
import numpy as np
import random
import config
from dag_server import *


# ([0-9]) # partiton size ([0-9])
# $2 # partiton size $1


class DAGDataSet:
    def __init__(self, num_timeslots=1, num_services=3, apply_partition='horizontal', graph_coarsening=False, layer_coarsening=False, net_manager=None, svc_arrival=None):
        self.num_timeslots = num_timeslots
        self.num_services = num_services
        self.apply_partition = apply_partition
        self.precision = 4 # bytes
        self.svc_set, self.system_manager = self.data_gen(net_manager=net_manager, svc_arrival=svc_arrival, apply_partition=apply_partition, layer_coarsening=layer_coarsening)
        if apply_partition == 'horizontal':
            self.coarsened_graph = self.horizontal_partitioning()
            self.graph = [np.unique(cg) for cg in self.coarsened_graph]
        else:
            self.coarsened_graph = [np.arange(len(svc.partitions)) for svc in self.svc_set.services]
            self.graph = [np.unique(cg) for cg in self.coarsened_graph]

        # 미리 계산이 필요한 정보들
        self.piece_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])
        self.piece_service_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])
        self.partition_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in cg])
        self.partition_service_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in cg])
        self.partition_arrival_map = np.array([self.svc_set.services[svc_id].arrival for svc_id in self.partition_service_map])
        self.partition_layer_map = np.array([self.svc_set.partitions[idx].layer_idx for idx in range(self.num_partitions)])
        if apply_partition == 'horizontal':
            self.partition_piece_map = np.array([self.svc_set.partitions[idx].piece_idx for idx in range(self.num_partitions)])
        else:
            self.partition_piece_map = np.arange(self.num_partitions)
        self.partition_workload_map = np.array([self.svc_set.partitions[idx].workload_size for idx in range(self.num_partitions)])
        self.partition_memory_map = np.array([self.svc_set.partitions[idx].memory for idx in range(self.num_partitions)])

        self.system_manager.piece_device_map = self.piece_device_map
        self.system_manager.piece_service_map = self.piece_service_map
        self.system_manager.partition_device_map = self.partition_device_map
        self.system_manager.partition_service_map = self.partition_service_map
        self.system_manager.partition_arrival_map = self.partition_arrival_map
        self.system_manager.partition_layer_map = self.partition_layer_map
        self.system_manager.partition_piece_map = self.partition_piece_map
        self.system_manager.partition_workload_map = self.partition_workload_map
        self.system_manager.partition_memory_map = self.partition_memory_map

        self.system_manager.calc_average()
        self.system_manager.calculate_schedule()

        # Apply multilevel graph partitioning algorithm
        # Return optimized DNN blocks
        if apply_partition == 'horizontal' and graph_coarsening == True:
            from mgp import MultilevelGraphPartitioning
            MGP = MultilevelGraphPartitioning(dataset=self, k=3)
            self.coarsened_graph = MGP.run_algo()
            self.graph = [np.unique(cg) for cg in self.coarsened_graph]
            self.num_pieces = sum([len(np.unique(cg)) for cg in self.coarsened_graph])
            start = end = unique_start = unique_end = 0
            for cg in self.coarsened_graph:
                for order, idx in enumerate(np.unique(cg)):
                    cg[np.where(cg == idx)] = order
                start = end
                end += len(cg)
                unique_start = unique_end
                unique_end += len(np.unique(cg))
                self.partition_piece_map[start:end] = cg + unique_start

    # Create random service arrival rate (almost deprecated)
    def create_arrival_rate(self, num_services, minimum, maximum):
        return minimum + (maximum - minimum) * np.random.random(num_services)

    # Piecewise partitioning for Convolutional layer
    def cnn_partitioning(self, layer_info, num_partitions, min_unit, piece_idx_start):
        min_unit_partitions = []
        output_data_location = []
        min_unit_start = min_unit_end = 0
        for idx in range(num_partitions):
            partition = deepcopy(layer_info)
            partition['layer_name'] = layer_info['layer_name'] + '_{:d}'.format(idx)
            partition['piece_idx'] = piece_idx_start + idx
            partition['original_layer_name'] = layer_info['layer_name']
            min_unit_start = min_unit_end
            min_unit_end += min_unit
            partition['output_height'] = min_unit_end - min_unit_start if idx < (num_partitions - 1) else layer_info['output_height'] - min_unit_start
            partition['input_height'] = (partition['output_height'] - 1) * layer_info['stride'] + partition['kernel'] - 1
            start = layer_info['stride'] * min_unit * idx - layer_info['padding']
            end = min(start + partition['input_height'], layer_info['input_height'] - 1)
            start = max(start, 0)
            partition['input_height'] = end - start
            # print(layer_info['layer_name'], "start:", start, "end:", end, '[kernel/2]:', math.floor(layer_info['kernel']/2), 'padding:', layer_info['padding'], "input:", partition['input_height'], "output:", partition['output_height'])
            partition['input_data_location'] = [i for i in range(start, end + 1)]
            if len(partition['predecessors']) > 0:
                partition['input_data_size'] = partition['input_height'] * partition['input_width'] * partition['input_channel'] * self.precision
            else:
                partition['input_data_size'] = partition['input_height'] * partition['input_width'] * partition['input_channel']
            partition['workload_size'] *= partition['output_height'] / layer_info['output_height']
            partition['memory'] *= partition['output_height'] / layer_info['output_height']
            output_data_location += [(partition['layer_name'], i) for i, _ in enumerate(range(partition['output_height']))]
            min_unit_partitions.append(partition)
        return min_unit_partitions, output_data_location

    # Piecewise partitioning for Fully-connected layer
    def fc_partitioning(self, layer_info, num_partitions, min_unit, piece_idx_start):
        min_unit_partitions = []
        output_data_location = []
        min_unit_start = min_unit_end = 0
        for idx in range(num_partitions):
            partition = deepcopy(layer_info)
            partition['layer_name'] = layer_info['layer_name'] + '_{:d}'.format(idx)
            partition['piece_idx'] = piece_idx_start + idx
            partition['original_layer_name'] = layer_info['layer_name']
            if layer_info['input_height'] > 1:
                min_unit_start = min_unit_end
                min_unit_end = min(min_unit_start + max(math.floor(layer_info['input_height'] / num_partitions), 1), layer_info['input_height'])
                partition['input_data_location'] = [i for i in range(min_unit_start, min_unit_end)]
            else:
                partition['input_height'] = 1
                partition['input_width'] = 1
                partition['input_channel'] = min_unit
            partition['workload_size'] /= num_partitions
            partition['memory'] /= num_partitions
            min_unit_partitions.append(partition)
            output_data_location += [(partition['layer_name'], 0), (partition['layer_name'], partition['output_channel']-1)]
        return min_unit_partitions, output_data_location

    def data_gen(self, net_manager=None, svc_arrival=None, apply_partition=None, layer_coarsening=False):
        self.service_info = [deepcopy(config.service_info[i]) for i in range(self.num_services)]
        system_manager = SystemManager()

        # create service set
        svc_set = ServiceSet()
        layer_idx_start = 0
        piece_idx_start = 0
        for dnn in self.service_info:
            svc = Service(dnn['model_name'], dnn['deadline'])
            
            # workload size calculation
            for layer_info in dnn['layers']:
                if layer_info['layer_type'] == 'cnn':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * layer_info['input_channel'] * layer_info['kernel'] * layer_info['kernel']
                    layer_info['memory'] = layer_info['kernel'] * layer_info['kernel'] * layer_info['input_channel'] * layer_info['output_channel'] * self.precision + layer_info['output_channel'] * self.precision
                    layer_info['memory'] += layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * self.precision
                elif layer_info['layer_type'] == 'maxpool' or layer_info['layer_type'] == 'avgpool':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * layer_info['kernel'] * layer_info['kernel']
                    layer_info['memory'] = layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * self.precision
                elif layer_info['layer_type'] == 'fc':
                    layer_info['workload_size'] *= layer_info['output_height'] * layer_info['output_width'] * layer_info['output_channel'] * (2 * layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] - 1)
                    layer_info['memory'] = layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * layer_info['output_channel'] * self.precision + layer_info['output_channel'] * self.precision
                    layer_info['memory'] += layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] * self.precision

            # just partitioning layers into minimum unit partitions
            if self.apply_partition:
                partitioned_layers = []
                for layer_idx, layer_info in enumerate(dnn['layers']):
                    layer_info['layer_idx'] = layer_idx + layer_idx_start
                    if layer_info['layer_type'] in ['cnn','maxpool']:
                        if dnn['model_name'] == 'GoogLeNet':
                            num_partitions = 3 # partiton size 6
                            min_unit = max(math.floor(layer_info['output_height'] / num_partitions), 1)
                            if layer_info['layer_name'] == 'conv1':
                                min_unit -= 1 # partiton size 0
                        elif dnn['model_name'] == 'ResNet-50':
                            num_partitions = 3 # partiton size 7
                            min_unit = max(math.floor(layer_info['output_height'] / num_partitions), 1)
                            if layer_info['layer_name'] == 'conv1':
                                min_unit -= 1 # partiton size 0
                        elif dnn['model_name'] == 'AlexNet':
                            num_partitions = 3 # partiton size 6
                            min_unit = max(math.floor(layer_info['output_height'] / num_partitions), 1)
                            if layer_info['layer_name'] == 'conv1':
                                min_unit -= 0 # partiton size 1
                        elif dnn['model_name'] == 'VGG-16':
                            num_partitions = 3 # partiton size 6
                            min_unit = max(math.floor(layer_info['output_height'] / num_partitions), 1)
                            if layer_info['layer_name'] == 'conv1':
                                min_unit -= 0 # partiton size 1
                        else:
                            min_unit = 1

                        min_unit_partitions, output_data_location = self.cnn_partitioning(layer_info, num_partitions=num_partitions, min_unit=min_unit, piece_idx_start=piece_idx_start)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'min_unit_partitions':min_unit_partitions, 'output_data_location':output_data_location})
                    elif layer_info['layer_type'] in ['fc','avgpool']:
                        if dnn['model_name'] == 'GoogLeNet':
                            num_partitions = 1 # partiton size 6
                            min_unit = max(math.floor(layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] / num_partitions), 1)
                        elif dnn['model_name'] == 'ResNet-50':
                            num_partitions = 1 # partiton size 7
                            min_unit = max(math.floor(layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] / num_partitions), 1)
                        elif dnn['model_name'] == 'AlexNet':
                            num_partitions = 1 # partiton size 6
                            min_unit = max(math.floor(layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] / num_partitions), 1)
                        elif dnn['model_name'] == 'VGG-16':
                            num_partitions = 1 # partiton size 6
                            min_unit = max(math.floor(layer_info['input_height'] * layer_info['input_width'] * layer_info['input_channel'] / num_partitions), 1)
                        else:
                            min_unit = 256

                        min_unit_partitions, output_data_location = self.fc_partitioning(layer_info, num_partitions=num_partitions, min_unit=min_unit, piece_idx_start=piece_idx_start)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'min_unit_partitions':min_unit_partitions, 'output_data_location':output_data_location})
                    else:
                        partitioned_layers.append(layer_info)
                piece_idx_start += num_partitions

                # calculate input/output slicing
                for layer_info in partitioned_layers:
                    start_input_index = 0
                    for partition in layer_info['min_unit_partitions']:
                        if 'input_data_location' in partition:
                            for pred_layer_name in partition['predecessors']:
                                pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                                partition['input_slicing'] = dict()
                                for i in partition['input_data_location']:
                                    if pred_layer['output_data_location'][i][0] in partition['input_slicing']:
                                        if partition['input_slicing'][pred_layer['output_data_location'][i][0]][0] > pred_layer['output_data_location'][i][1]:
                                            partition['input_slicing'][pred_layer['output_data_location'][i][0]][0] = pred_layer['output_data_location'][i][1]
                                        elif partition['input_slicing'][pred_layer['output_data_location'][i][0]][1] < pred_layer['output_data_location'][i][1]:
                                            partition['input_slicing'][pred_layer['output_data_location'][i][0]][1] = pred_layer['output_data_location'][i][1]
                                    else:
                                        partition['input_slicing'][pred_layer['output_data_location'][i][0]] = [pred_layer['output_data_location'][i][1], pred_layer['output_data_location'][i][1]]
                            if len(partition['predecessors']) < 1:
                                partition['input_slicing'] = dict()
                                partition['input_slicing']['input'] = [partition['input_data_location'][0], partition['input_data_location'][-1]]
                        else: # fc, avgpool exception
                            partition['input_slicing'] = dict()
                            for pred_layer_name in partition['predecessors']:
                                pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                                for (layer_name, location) in pred_layer['output_data_location']:
                                    if layer_name in partition['input_slicing']:
                                        if partition['input_slicing'][layer_name][0] > location:
                                            partition['input_slicing'][layer_name][0] = location
                                        elif partition['input_slicing'][layer_name][1] < location:
                                            partition['input_slicing'][layer_name][1] = location
                                    else:
                                        partition['input_slicing'][layer_name] = [location, location]

                # predecessor recalculation for the minimum unit partitions
                partitions = []
                for layer_info in partitioned_layers:
                    if layer_info['layer_type'] in ['cnn','maxpool']:
                        for partition in layer_info['min_unit_partitions']:
                            predecessors = []
                            input_data_size = []
                            for pred_layer_name in partition['predecessors']:
                                pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                                for input_location in partition['input_data_location']:
                                    # find predecessor partition and calculate input size from the predecessor
                                    pred_partition = next(l for l in pred_layer['min_unit_partitions'] if l['layer_name'] == pred_layer['output_data_location'][input_location][0])
                                    if pred_partition['layer_name'] not in predecessors:
                                        predecessors.append(pred_partition['layer_name'])
                                        input_data_size.append(partition['input_width'] * partition['input_channel'] * self.precision)
                                    else:
                                        input_data_size[predecessors.index(pred_partition['layer_name'])] += partition['input_width'] * partition['input_channel'] * self.precision
                            partition['predecessors'] = predecessors
                            if len(predecessors) > 0 :
                                partition['input_data_size'] = input_data_size
                            partition['successors'] = []
                            partition['output_data_size'] = []
                            del partition['input_data_location']
                            partitions.append(partition)
                    elif layer_info['layer_type'] == 'avgpool' or (layer_info['layer_type'] == 'fc' and layer_info['min_unit_partitions'][0]['is_first_fc'] == True):
                        for partition in layer_info['min_unit_partitions']:
                            predecessors = []
                            input_data_size = []
                            for pred_layer_name in partition['predecessors']:
                                pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                                for pred_partition in pred_layer['min_unit_partitions']:
                                    # if pred_partition['piece_idx'] == partition['piece_idx']: # piece idx가 아니라 output input matching 이어야하는데
                                    predecessors.append(pred_partition['layer_name'])
                                    input_data_size.append(pred_partition['output_height'] * pred_partition['output_width'] * pred_partition['output_channel'] * self.precision)
                            if len(predecessors) > 0:
                                partition['predecessors'] = predecessors
                                partition['input_data_size'] = input_data_size
                            partition['successors'] = []
                            partition['output_data_size'] = []
                            partitions.append(partition)
                    elif layer_info['layer_type'] == 'fc' and layer_info['min_unit_partitions'][0]['is_first_fc'] == False:
                        for partition in layer_info['min_unit_partitions']:
                            predecessors = []
                            input_data_size = []
                            for pred_layer_name in partition['predecessors']:
                                pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                                for pred_partition in pred_layer['min_unit_partitions']:
                                    predecessors.append(pred_partition['layer_name'])
                                    input_data_size.append(pred_partition['output_height'] * pred_partition['output_width'] * pred_partition['output_channel'] * self.precision)
                            if len(predecessors) > 0:
                                partition['predecessors'] = predecessors
                                partition['input_data_size'] = input_data_size
                            partition['successors'] = []
                            partition['output_data_size'] = []
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
                    svc.partitions.append(Partition(svc_set=svc_set, service=svc, **partition))

                # print(dnn['model_name']) # for piecewise partition debug
                # for partition in partitions:
                #     print(partition)
                # input()
            else:
                for layer_idx, layer_info in enumerate(dnn['layers']):
                    layer_info['layer_idx'] = layer_idx + layer_idx_start
                    svc.partitions.append(Partition(svc_set=svc_set, service=svc, **layer_info))
            layer_idx_start += len(dnn['layers'])

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
                svc.partition_predecessor[partition.id] = [pred.id for pred in partition.predecessors]
                svc.partition_successor[partition.id] = [succ.id for succ in partition.successors]
                svc_set.partition_predecessor[partition.total_id] = [pred.total_id for pred in partition.predecessors]
                svc_set.partition_successor[partition.total_id] = [succ.total_id for succ in partition.successors]
            
            # predecessor, successor data size check
            for partition in svc.partitions:
                if len(partition.predecessors) > 0:
                    for i in range(len(partition.predecessors)):
                        svc.input_data_size[(partition.predecessors[i].id,partition.id)] = partition.input_data_size[i]
                        svc_set.input_data_size[(partition.predecessors[i].total_id,partition.total_id)] = partition.input_data_size[i]
                else:
                    svc.input_data_size[(partition.id,partition.id)] = partition.input_data_size
                    svc_set.input_data_size[(partition.total_id,partition.total_id)] = partition.input_data_size
                if len(partition.successors) > 0:
                    for i in range(len(partition.successors)):
                        svc.output_data_size[(partition.id,partition.successors[i].id)] = partition.output_data_size[i]
                        svc_set.output_data_size[(partition.total_id,partition.successors[i].total_id)] = partition.output_data_size[i]
                else:
                    svc.output_data_size[(partition.id,partition.id)] = partition.output_data_size
                    svc_set.output_data_size[(partition.total_id,partition.total_id)] = partition.output_data_size

            # dag structure error check
            for p1 in svc.partitions:
                for p2 in svc.partitions:
                    if ((p1.id,p2.id) in svc.input_data_size.keys() and (p2.id,p1.id) in svc.output_data_size.keys()) and svc.input_data_size[(p1.id,p2.id)] != svc.output_data_size[(p2.id,p1.id)]:
                        raise RuntimeError("DAG input output data mismatch!!", (p1.layer_name, p2.layer_name))

        self.num_services = len(svc_set.services)
        self.num_partitions = len(svc_set.partitions)
        print("self.num_partitions", self.num_partitions)


        # create arrival rate table
        self.max_arrival = 1
        self.min_arrival = 1

        if svc_arrival is None:
            svc_arrival = list()
            for t in range(self.num_timeslots):
                svc_arrival.append(self.create_arrival_rate(num_services=self.num_services, minimum=self.min_arrival, maximum=self.max_arrival))
            svc_arrival = np.array(svc_arrival)

        # create servers
        request = dict()
        local = dict()
        edge = dict()
        cloud = dict()

        self.num_requests = len(config.request_device_info)
        self.num_locals = len(config.local_device_info)
        self.num_edges = len(config.edge_server_info)
        self.num_clouds = len(config.cloud_server_info)
        self.num_servers = self.num_requests + self.num_locals + self.num_edges + self.num_clouds
        id = 0
        for request_device in config.request_device_info:
            request_device = deepcopy(request_device)
            request_device['computing_intensity'] = request_device['computing_intensity']
            request[id] = Server(**request_device, system_manager=system_manager, id=id)
            id += 1
        for local_device in config.local_device_info:
            local_device = deepcopy(local_device)
            local_device['computing_intensity'] = local_device['computing_intensity']
            local[id] = Server(**local_device, system_manager=system_manager, id=id)
            id += 1
        for edge_server in config.edge_server_info:
            edge_server = deepcopy(edge_server)
            edge_server['computing_intensity'] = edge_server['computing_intensity']
            edge[id] = Server(**edge_server, system_manager=system_manager, id=id)
            id += 1
        for cloud_server in config.cloud_server_info:
            cloud_server = deepcopy(cloud_server)
            cloud_server['computing_intensity'] = cloud_server['computing_intensity']
            cloud[id] = Server(**cloud_server, system_manager=system_manager, id=id)
            id += 1

        # create network manager
        if net_manager == None:
            net_manager = NetworkManager(channel_bandwidth=1024*1024*100, gaussian_noise=1, B_edge_up=1024*1024*100, B_edge_down=1024*1024*100, B_cloud_up=1024*1024*1024, B_cloud_down=1024*1024*1024, request=request, local=local, edge=edge, cloud=cloud)
            net_manager.P_d = np.ones(shape=(self.num_servers,))
            net_manager.g_dd = np.zeros(shape=(self.num_servers, self.num_servers))
            for i in range(self.num_servers):
                for j in range(i + 1, self.num_servers):
                    net_manager.g_dd[i, j] = net_manager.g_dd[j, i] = 1 # np.random.normal(1.0, 0.2) # random.uniform(1.0, 1.0)
                net_manager.g_dd[i, i] = 0
            net_manager.cal_b_dd()
            print(net_manager.B_dd)
            print(net_manager.g_dd)
            input()

        # init system manager
        system_manager.net_manager = net_manager
        system_manager.num_timeslots = self.num_timeslots
        system_manager.num_servers = self.num_servers
        system_manager.num_services = self.num_services
        system_manager.num_partitions = self.num_partitions
        system_manager.set_service_set(svc_set, svc_arrival)
        system_manager.set_servers(request, local, edge, cloud)
        return svc_set, system_manager

    def horizontal_partitioning(self):
        coarsened_graph = []
        start = 0
        end = 0
        for svc in self.svc_set.services:
            num_partitions = len(svc.partitions)
            cg = np.arange(num_partitions)
            start = end
            end = start + num_partitions

            horizontal_partition = dict()
            if self.service_info[svc.id]['model_name'] == 'GoogLeNet':
                num_neuron_partitions = 3 # partiton size 6
            elif self.service_info[svc.id]['model_name'] == 'ResNet-50':
                num_neuron_partitions = 3 # partiton size 7
            elif self.service_info[svc.id]['model_name'] == 'AlexNet':
                num_neuron_partitions = 3 # partiton size 6
            elif self.service_info[svc.id]['model_name'] == 'VGG-16':
                num_neuron_partitions = 3 # partiton size 6

            for i in range(num_neuron_partitions):
                horizontal_partition[i] = []
            for l in self.service_info[svc.id]['layers']:
                p_lst = [p.layer_name for p in svc.partitions if p.original_layer_name == l['layer_name']]
                for i in range(num_neuron_partitions):
                    start = math.floor(len(p_lst) / num_neuron_partitions * i)
                    end = math.floor(len(p_lst) / num_neuron_partitions * (i + 1))
                    for j in range(start, end):
                        horizontal_partition[i].append(l['layer_name']+'_'+str(j))

            for p_id in range(num_partitions):
                for i in range(num_neuron_partitions):
                    if svc.partitions[p_id].layer_name in horizontal_partition[i]:
                        cg[p_id] = i

            coarsened_graph.append(cg)
        return coarsened_graph