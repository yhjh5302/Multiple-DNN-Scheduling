from common import *
from models import AlexNet
from dag_data_generator import DAGDataSet
from algorithms.Greedy import HEFT, Greedy


server_mapping = {0: 1, 1: 0}


def scheduler(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, proc_schedule_list, proc_schedule_lock, _stop_event):
    with open("outputs/net_manager_backup", "rb") as fp:
        net_manager = pickle.load(fp)
    dataset = DAGDataSet(num_timeslots=1, num_services=1, net_manager=net_manager, apply_partition="horizontal", graph_coarsening=True)
    num_servers = 1
    algorithm = Greedy(dataset=dataset)
    algorithm.rank = "rank_d"
    algorithm.server_lst = list(dataset.system_manager.request.keys())[:num_servers] + list(dataset.system_manager.edge.keys())

    tag = 0
    p_tag = 0
    partitions = dataset.system_manager.service_set.partitions
    while _stop_event.is_set() == False:
        # request를 반복적으로 받음
        input_src = recv_request()
        # scheduling 수행
        (([server], [order]), [latency], took) = algorithm.run_algo()
        # partition p를 순서대로
        start = time.time()
        for p_id in order:
            p = partitions[p_id]
            # p가 받아야할 input 데이터들에 대해서
            for i, (pred, slicing_index) in enumerate(p.input_slicing.items()):
                if i == 0:
                    proc_flag = True
                else:
                    proc_flag = False
                if pred == 'input':
                    src = input_src
                    pred_id = -1
                else:
                    src = server_mapping[server[next(i for i, l in enumerate(partitions) if l.layer_name == pred)]]
                    pred_id = next(i for i, l in enumerate(partitions) if l.layer_name == pred)
                print("pred_id", pred_id, p_id)
                dst = server_mapping[server[p_id]]
                schedule = torch.tensor([dataset.partition_layer_map[p_id], len(p.input_slicing), len(p.successors), p_tag+pred_id, p_tag+p_id, src, dst, p.input_height, p.input_width, p.input_channel, slicing_index[0], slicing_index[1], tag, proc_flag], dtype=torch.int32)
                # dst는 데이터를 받는 역할을 함
                # 데이터의 dst에 스케줄 보냄
                if dst == 0:
                    if schedule[13] == True:
                        with proc_schedule_lock:
                            proc_schedule_list.append(schedule)
                    schedule[5] = src
                    schedule[6] = -1
                    with recv_schedule_lock:
                        recv_schedule_list.append(schedule.clone())
                else:
                    schedule[5] = src
                    schedule[6] = -1
                    send_schedule(schedule=schedule.clone(), dst=dst)
                # src는 데이터를 보내는 역할을 함
                # 데이터의 src에 스케줄 보냄
                if src == 0:
                    schedule[5] = -1
                    schedule[6] = dst
                    with send_schedule_lock:
                        send_schedule_list.append(schedule.clone())
                else:
                    schedule[5] = -1
                    schedule[6] = dst
                    send_schedule(schedule=schedule.clone(), dst=src)

                del schedule
                tag += 1
        p_tag += len(order)
        print("scheduling took", time.time() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Piecewise Partition and Scheduling')
    parser.add_argument('--vram_limit', default=0.2, type=float, help='GPU memory limit')
    parser.add_argument('--master_addr', default='localhost', type=str, help='Master node ip address')
    parser.add_argument('--master_port', default='30000', type=str, help='Master node port')
    parser.add_argument('--rank', default=0, type=int, help='Master node port', required=True)
    parser.add_argument('--data_path', default='./Data/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--roi_name', default='roi.jpg', type=str, help='RoI file name')
    parser.add_argument('--num_nodes', default=2, type=int, help='Number of nodes')
    parser.add_argument('--resolution', default=(854, 480), type=tuple, help='Image resolution')
    parser.add_argument('--verbose', default=False, type=str2bool, help='If you want to print debug messages, set True')
    args = parser.parse_args()

    # gpu setting
    # torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(fraction=args.vram_limit, device=device)
    print(device, torch.cuda.get_device_name(0))

    # model loading
    model = AlexNet().eval()

    # cluster connection setup
    print('Waiting for the cluster connection...')
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group('gloo', rank=args.rank, world_size=args.num_nodes)

    # data sender/receiver thread start
    _stop_event = threading.Event()
    recv_data_queue = queue.PriorityQueue()
    recv_data_lock = threading.Lock()
    send_data_list = []
    send_data_lock = threading.Lock()
    internal_data_list = []
    internal_data_lock = threading.Lock()
    send_schedule_list = []
    send_schedule_lock = threading.Lock()
    recv_schedule_list = []
    recv_schedule_lock = threading.Lock()
    proc_schedule_list = []
    proc_schedule_lock = threading.Lock()

    threading.Thread(target=scheduler, args=(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, proc_schedule_list, proc_schedule_lock, _stop_event)).start()
    threading.Thread(target=recv_thread, args=(args.rank, recv_schedule_list, recv_schedule_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event)).start()
    threading.Thread(target=send_thread, args=(args.rank, send_schedule_list, send_schedule_lock, send_data_list, send_data_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event)).start()

    while _stop_event.is_set() == False:
        inputs, layer_id, p_id, num_outputs = bring_data(recv_data_queue, recv_data_lock, proc_schedule_list, proc_schedule_lock, _stop_event)
        outputs = model(inputs, layer_id)
        print("processing_done", outputs.shape)
        with send_data_lock:
            send_data_list.append((p_id, num_outputs, outputs))