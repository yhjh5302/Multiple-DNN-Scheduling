from copy import deepcopy
import numpy as np
import random
import config
from dag_server import *
from multilevel_graph_partitioning import MultiLevelGraphPartitioning


class DAGDataSet:
    def __init__(self, num_timeslots=1, num_services=3, apply_partition=True, net_manager=None, svc_arrival=None):
        self.num_timeslots = num_timeslots
        self.num_services = num_services
        self.apply_partition = apply_partition
        self.svc_set, self.system_manager = self.data_gen(net_manager=net_manager, svc_arrival=svc_arrival)
        if apply_partition:
            self.coarsened_graph = self.CoEdge_partitioning()
        else:
            self.coarsened_graph = [np.arange(len(svc.partitions)) for svc in self.svc_set.services]
        self.partition_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in np.unique(cg)])
        self.system_manager.partition_device_map = np.array([idx for idx, cg in enumerate(self.coarsened_graph) for _ in cg])

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
        self.service_info = [deepcopy(config.service_info[i]) for i in range(self.num_services)]
        system_manager = SystemManager()

        # create service set
        svc_set = ServiceSet()
        for dnn in self.service_info:
            svc = Service(dnn['model_name'], dnn['deadline'])
            
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
                    if layer_info['layer_type'] in ['cnn','maxpool','avgpool']:
                        if dnn['model_name'] == 'GoogLeNet':
                            min_unit = max(math.floor(layer_info['output_height'] / 6), 1)
                        elif dnn['model_name'] == 'AlexNet':
                            min_unit = max(math.floor(layer_info['output_height'] / 6), 1)
                        elif dnn['model_name'] == 'ResNet-50':
                            min_unit = max(math.floor(layer_info['output_height'] / 7), 1)
                        else:
                            min_unit = 1

                        min_unit_partitions, output_data_location = self.cnn_partitioning(layer_info, min_unit=min_unit)
                        partitioned_layers.append({'layer_name':layer_info['layer_name'], 'layer_type':layer_info['layer_type'], 'min_unit_partitions':min_unit_partitions, 'output_data_location':output_data_location})
                    elif layer_info['layer_type'] == 'fc':
                        partitioned_layers.append(layer_info)
                    else:
                        partitioned_layers.append(layer_info)
                # predecessor recalculation for the minimum unit partitions
                partitions = []
                for layer_info in partitioned_layers:
                    if layer_info['layer_type'] in ['cnn','maxpool','avgpool']:
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
                        input_data_size = []
                        for pred_layer_name in layer_info['predecessors']:
                            pred_layer = next(l for l in partitioned_layers if l['layer_name'] == pred_layer_name)
                            if pred_layer['layer_type'] in ['cnn','maxpool','avgpool']:
                                for pred_partition in pred_layer['min_unit_partitions']:
                                    predecessors.append(pred_partition['layer_name'])
                                    input_data_size.append(pred_partition['output_height'] * pred_partition['output_width'] * pred_partition['output_channel'] * 4)
                            else:
                                break
                        if len(predecessors) > 0:
                            layer_info['predecessors'] = predecessors
                            layer_info['input_data_size'] = input_data_size
                        partitions.append(layer_info)
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
            net_manager = NetworkManager(channel_bandwidth=1024*1024*15, channel_gain=1, gaussian_noise=1, B_edge_up=1024*1024*10, B_edge_down=1024*1024*40, B_cloud_up=1024*1024*1024, B_cloud_down=1024*1024*1024, request=request, local=local, edge=edge, cloud=cloud)
            net_manager.P_dd = np.zeros(shape=(self.num_servers, self.num_servers))
            for i in range(self.num_servers):
                for j in range(i + 1, self.num_servers):
                    net_manager.P_dd[i, j] = net_manager.P_dd[j, i] = random.uniform(0.666, 1)
                net_manager.P_dd[i, i] = 0
            net_manager.cal_b_dd()

        # init system manager
        system_manager.net_manager = net_manager
        system_manager.num_timeslots = self.num_timeslots
        system_manager.num_servers = self.num_servers
        system_manager.num_services = self.num_services
        system_manager.num_partitions = self.num_partitions
        system_manager.set_service_set(svc_set, svc_arrival)
        system_manager.set_servers(request, local, edge, cloud)

        system_manager.rank_u = np.zeros(self.num_partitions)
        system_manager.calc_average()
        for svc in svc_set.services:
            for partition in svc.partitions:
                system_manager.calc_rank_u(partition)

        return svc_set, system_manager

    def CoEdge_partitioning(self):
        coarsened_graph = []
        start = 0
        end = 0
        for svc in self.svc_set.services:
            num_partitions = len(svc.partitions)
            cg = np.arange(num_partitions)
            start = end
            end = start + num_partitions

            coedge_partition = dict()
            if self.service_info[svc.id]['model_name'] == 'GoogLeNet':
                for i in range(7):
                    coedge_partition[i] = []
                for l in self.service_info[svc.id]['layers']:
                    p_lst = [p.layer_name for p in svc.partitions if (p.layer_type in ['cnn','maxpool','avgpool'] and p.original_layer_name == l['layer_name']) or (p.layer_type == 'fc' and p.layer_name == l['layer_name'])]
                    if l['layer_type'] in ['cnn','maxpool','avgpool']:
                        for i in range(6):
                            start = math.floor(len(p_lst) / 6 * i)
                            end = math.floor(len(p_lst) / 6 * (i + 1))
                            for j in range(start, end):
                                coedge_partition[i].append(l['layer_name']+'_'+str(j))
                    elif l['layer_type'] == 'fc':
                        coedge_partition[6].extend(p_lst)

                for p_id in range(num_partitions):
                    for i in range(7):
                        if svc.partitions[p_id].layer_name in coedge_partition[i]:
                            cg[p_id] = i

            elif self.service_info[svc.id]['model_name'] == 'AlexNet':
                for i in range(7):
                    coedge_partition[i] = []
                for l in self.service_info[svc.id]['layers']:
                    p_lst = [p.layer_name for p in svc.partitions if (p.layer_type in ['cnn','maxpool','avgpool'] and p.original_layer_name == l['layer_name']) or (p.layer_type == 'fc' and p.layer_name == l['layer_name'])]
                    if l['layer_type'] in ['cnn','maxpool','avgpool']:
                        for i in range(6):
                            start = math.floor(len(p_lst) / 6 * i)
                            end = math.floor(len(p_lst) / 6 * (i + 1))
                            for j in range(start, end):
                                coedge_partition[i].append(l['layer_name']+'_'+str(j))
                    elif l['layer_type'] == 'fc':
                        coedge_partition[6].extend(p_lst)

                for p_id in range(num_partitions):
                    for i in range(7):
                        if svc.partitions[p_id].layer_name in coedge_partition[i]:
                            cg[p_id] = i

            elif self.service_info[svc.id]['model_name'] == 'ResNet-50':
                for i in range(8):
                    coedge_partition[i] = []
                for l in self.service_info[svc.id]['layers']:
                    p_lst = [p.layer_name for p in svc.partitions if (p.layer_type in ['cnn','maxpool','avgpool'] and p.original_layer_name == l['layer_name']) or (p.layer_type == 'fc' and p.layer_name == l['layer_name'])]
                    if l['layer_type'] in ['cnn','maxpool','avgpool']:
                        for i in range(7):
                            start = math.floor(len(p_lst) / 7 * i)
                            end = math.floor(len(p_lst) / 7 * (i + 1))
                            for j in range(start, end):
                                coedge_partition[i].append(l['layer_name']+'_'+str(j))
                    elif l['layer_type'] == 'fc':
                        coedge_partition[7].extend(p_lst)

                for p_id in range(num_partitions):
                    for i in range(8):
                        if svc.partitions[p_id].layer_name in coedge_partition[i]:
                            cg[p_id] = i

            elif self.service_info[svc.id]['model_name'] == 'VGG-F':
                for i in range(8):
                    coedge_partition[i] = []
                for l in self.service_info[svc.id]['layers']:
                    p_lst = [p.layer_name for p in svc.partitions if (p.layer_type in ['cnn','maxpool','avgpool'] and p.original_layer_name == l['layer_name']) or (p.layer_type == 'fc' and p.layer_name == l['layer_name'])]
                    if l['layer_type'] in ['cnn','maxpool','avgpool']:
                        for i in range(7):
                            start = math.floor(len(p_lst) / 7 * i)
                            end = math.floor(len(p_lst) / 7 * (i + 1))
                            for j in range(start, end):
                                coedge_partition[i].append(l['layer_name']+'_'+str(j))
                    elif l['layer_type'] == 'fc':
                        coedge_partition[7].extend(p_lst)

                for p_id in range(num_partitions):
                    for i in range(8):
                        if svc.partitions[p_id].layer_name in coedge_partition[i]:
                            cg[p_id] = i
            coarsened_graph.append(cg)
        return coarsened_graph