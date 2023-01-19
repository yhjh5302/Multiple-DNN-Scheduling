from common import *
from dag_data_generator import DAGDataSet
from algorithms.Greedy import HEFT


def scheduler(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event):
    with open("outputs/net_manager_backup", "rb") as fp:
        net_manager = pickle.load(fp)
    dataset = DAGDataSet(num_timeslots=1, num_services=1, net_manager=net_manager, apply_partition="horizontal", graph_coarsening=True)
    algorithm = HEFT(dataset=dataset)
    algorithm.rank = "rank_u"
    (([x], [y]), [latency], took) = algorithm.run_algo()
    print(x, y, latency, took)

    ## for test
    for partition in dataset.system_manager.service_set.partitions:
        if hasattr(partition, 'input_slicing'):
            print(partition.layer_name, partition.input_slicing)
            for slicing in partition.input_slicing.values():
                print("[{}:{},:,:]".format(slicing[0], slicing[1]))
    return 

def processing(inputs, model):
    outputs = model(inputs)
    return outputs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Piecewise Partition and Scheduling')
    parser.add_argument('--vram_limit', default=0.2, type=float, help='GPU memory limit')
    parser.add_argument('--master_addr', default='192.168.1.2', type=str, help='Master node ip address')
    parser.add_argument('--master_port', default='30000', type=str, help='Master node port')
    parser.add_argument('--rank', default=0, type=int, help='Master node port', required=True)
    parser.add_argument('--data_path', default='./Data/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--roi_name', default='roi.jpg', type=str, help='RoI file name')
    parser.add_argument('--num_proc', default=2, type=int, help='Number of processes')
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_per_process_memory_fraction(fraction=args.vram_limit, device=device)
    print(device, torch.cuda.get_device_name(0))

    # cluster connection setup
    print('Waiting for the cluster connection...')
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    dist.init_process_group('gloo', init_method='tcp://%s:%s' % (args.master_addr, args.master_port), rank=args.rank, world_size=args.num_proc)

    # data sender/receiver thread start
    _stop_event = threading.Event()
    recv_data_list = []
    recv_data_lock = threading.Lock()
    send_data_list = []
    send_data_lock = threading.Lock()
    # recv_schedule_list = []
    recv_schedule_list = [[] for i in range(QUEUE_LENGTH)]
    recv_schedule_lock = threading.Lock()
    # send_schedule_list = []
    send_schedule_list = [[] for i in range(QUEUE_LENGTH)]
    send_schedule_lock = threading.Lock()

    threading.Thread(target=scheduler, args=(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event)).start()
    threading.Thread(target=recv_thread, args=(recv_schedule_list, recv_schedule_lock, recv_data_list, recv_data_lock, _stop_event)).start()
    threading.Thread(target=send_thread, args=(send_schedule_list, send_schedule_lock, send_data_list, send_data_lock, _stop_event)).start()

    while _stop_event.is_set() == False:
        inputs = bring_data(recv_data_list, recv_data_lock, _stop_event)
        outputs = processing(inputs)
        with send_data_lock:
            send_data_list.append(outputs)