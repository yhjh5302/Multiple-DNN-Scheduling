from common import *
from dag_data_generator import DAGDataSet
from algorithms.Greedy import HEFT, Greedy


num_pieces = 3
tag = 1
schedule_shape = ['src', 'input_height', 'input_width', 'input_channel', 'slicing_start', 'slicing_end', 'tag']
server_mapping = {19: 0, 12:1}


def schedule_packing(server, order, dataset): # 포맷 정해서 정형으로 tensor 만들고 전송.
    global tag
    schedule = torch.empty(size=(len(server_mapping), 2, dataset.num_partitions, len(schedule_shape)), dtype=torch.int32)
    for i, p_id in enumerate(order):
        tag += 1
        p = dataset.system_manager.service_set.partitions[p_id]
        for k in p.input_slicing:
            schedule[server_mapping[server[i]], 0, p_id] = torch.Tensor([server[i], p.input_height, p.input_width, p.input_channel, p.input_slicing[k][0], p.input_slicing[k][1], tag])
            schedule[server_mapping[server[i]], 1, p_id] = torch.Tensor([server[i], p.input_height, p.input_width, p.input_channel, p.input_slicing[k][0], p.input_slicing[k][1], tag])
        print(i, schedule[server_mapping[server[i]], 0, p_id])
    return schedule


def scheduler(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event):
    with open("outputs/net_manager_backup", "rb") as fp:
        net_manager = pickle.load(fp)
    dataset = DAGDataSet(num_timeslots=1, num_services=1, net_manager=net_manager, apply_partition="horizontal", graph_coarsening=True)
    algorithm = Greedy(dataset=dataset)

    while _stop_event.is_set() == False:
        (([server], [order]), [latency], took) = algorithm.run_algo()
        schedule_list = schedule_packing(server, order, dataset)
        for dst, schedule in enumerate(schedule_list):
            send_schedule(schedule=schedule, dst=dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Piecewise Partition and Scheduling')
    parser.add_argument('--vram_limit', default=0.2, type=float, help='GPU memory limit')
    parser.add_argument('--master_addr', default='192.168.1.2', type=str, help='Master node ip address')
    parser.add_argument('--master_port', default='30000', type=str, help='Master node port')
    parser.add_argument('--rank', default=0, type=int, help='Master node port', required=True)
    parser.add_argument('--data_path', default='./Data/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--roi_name', default='roi.jpg', type=str, help='RoI file name')
    parser.add_argument('--num_nodes', default=2, type=int, help='Number of nodes')
    parser.add_argument('--resolution', default=(854, 480), type=tuple, help='Image resolution')
    parser.add_argument('--verbose', default=False, type=str2bool, help='If you want to print debug messages, set True')
    args = parser.parse_args()

    ### debug test
    _stop_event = threading.Event()
    recv_data_list = []
    recv_data_lock = threading.Lock()
    send_data_list = []
    send_data_lock = threading.Lock()
    recv_schedule_list = [[] for i in range(QUEUE_LENGTH)]
    recv_schedule_lock = threading.Lock()
    send_schedule_list = [[] for i in range(QUEUE_LENGTH)]
    send_schedule_lock = threading.Lock()
    threading.Thread(target=scheduler, args=(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event)).start()
    exit()

    # gpu setting
    # torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_per_process_memory_fraction(fraction=args.vram_limit, device=device)
    print(device, torch.cuda.get_device_name(0))

    # model loading
    model = VGGNet().eval()

    # cluster connection setup
    print('Waiting for the cluster connection...')
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group('gloo', rank=args.rank, world_size=args.num_nodes)

    # data sender/receiver thread start
    _stop_event = threading.Event()
    recv_data_list = []
    recv_data_lock = threading.Lock()
    send_data_list = []
    send_data_lock = threading.Lock()
    recv_schedule_list = [[] for i in range(QUEUE_LENGTH)]
    recv_schedule_lock = threading.Lock()
    send_schedule_list = [[] for i in range(QUEUE_LENGTH)]
    send_schedule_lock = threading.Lock()

    threading.Thread(target=scheduler, args=(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event)).start()
    threading.Thread(target=recv_thread, args=(recv_schedule_list, recv_schedule_lock, recv_data_list, recv_data_lock, _stop_event)).start()
    threading.Thread(target=send_thread, args=(send_schedule_list, send_schedule_lock, send_data_list, send_data_lock, _stop_event)).start()

    while _stop_event.is_set() == False:
        inputs = bring_data(recv_data_list, recv_data_lock, _stop_event)
        outputs = model(inputs)
        with send_data_lock:
            send_data_list.append(outputs)