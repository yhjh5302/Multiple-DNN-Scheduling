from common import *
from models import AlexNet
import cv2, logging



def data_generator(args, send_data_queue, send_data_lock):
    # video data load
    vid = cv2.VideoCapture(args.data_path+args.video_name)
    fps = vid.get(cv2.CAP_PROP_FPS)
    delay = 10000 # int(600/fps)
    roi_mask = cv2.imread(args.data_path+args.roi_name, cv2.IMREAD_UNCHANGED)
    roi_mask = cv2.resize(roi_mask, args.resolution, interpolation=cv2.INTER_CUBIC)

    kernel = None
    backgroundObject = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=128, detectShadows=False)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(size=(227,227),interpolation=0), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    while vid.isOpened():
        _, frame = vid.read()

        if frame is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, args.resolution, interpolation=cv2.INTER_CUBIC)
        detected = False

        # calculate the foreground mask
        took = time.time()
        foreground_mask = cv2.bitwise_and(frame, frame, mask=roi_mask)
        foreground_mask = backgroundObject.apply(foreground_mask)
        _, foreground_mask = cv2.threshold(foreground_mask, 250, 255, cv2.THRESH_BINARY)
        foreground_mask = cv2.erode(foreground_mask, kernel, iterations=1)
        foreground_mask = cv2.dilate(foreground_mask, kernel, iterations=10)
        if args.verbose:
            print("mask {:.5f} ms".format((time.time() - took) * 1000))

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxedFrame = frame.copy()
        # loop over each contour found in the frame.
        for cnt in contours:
            # We need to be sure about the area of the contours i.e. it should be higher than 256 to reduce the noise.
            if cv2.contourArea(cnt) > 256:
                detected = True
                # Accessing the x, y and height, width of the objects
                x, y, w, h = cv2.boundingRect(cnt)
                # Here we will be drawing the bounding box on the objects
                cv2.rectangle(boxedFrame, (x , y), (x + w, y + h),(0, 0, 255), 2)
                # Then with the help of putText method we will write the 'detected' on every object with a bounding box
                cv2.putText(boxedFrame, 'Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv2.LINE_AA)

        if args.verbose:
            # for_show_all_frames = np.hstack((frame, foreground_mask, boxedFrame))
            foregroundPart = cv2.bitwise_and(frame, frame, mask=foreground_mask)
            cv2.imshow('foregroundPart', foregroundPart)
            cv2.imshow('boxedFrame', boxedFrame)

        # send image info to the master and recv scheduling decision
        p_tag = send_request()
        with send_data_lock:
            send_data_queue.put((p_tag-1, num_pieces, transform(boxedFrame).unsqueeze(0)))
            print("send data", p_tag-1)
            p_tag += num_partitions + 3

        import random
        time.sleep(random.random()*5)

        if cv2.waitKey(delay) == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Piecewise Partition and Scheduling')
    parser.add_argument('--vram_limit', default=0.2, type=float, help='GPU memory limit')
    parser.add_argument('--master_addr', default='localhost', type=str, help='Master node ip address')
    parser.add_argument('--master_port', default='30000', type=str, help='Master node port')
    parser.add_argument('--rank', default=0, type=int, help='Master node port', required=True)
    parser.add_argument('--data_path', default='/home/jin/git/DNN/Data/AIC22_Track1_MTMC_Tracking/train/S03/c011/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--roi_name', default='roi.jpg', type=str, help='RoI file name')
    parser.add_argument('--num_nodes', default=5, type=int, help='Number of nodes')
    parser.add_argument('--resolution', default=(854, 480), type=tuple, help='Image resolution')
    parser.add_argument('--verbose', default=False, type=str2bool, help='If you want to print debug messages, set True')
    args = parser.parse_args()

    # gpu setting
    # torch.backends.cudnn.benchmark = True
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # torch.cuda.set_per_process_memory_fraction(fraction=args.vram_limit, device=device)
    # print(device, torch.cuda.get_device_name(0))

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
    send_data_queue = queue.PriorityQueue()
    send_data_lock = threading.Lock()
    internal_data_list = []
    internal_data_lock = threading.Lock()
    send_schedule_list = []
    send_schedule_lock = threading.Lock()
    recv_schedule_list = []
    recv_schedule_lock = threading.Lock()
    proc_schedule_list = []
    proc_schedule_lock = threading.Lock()

    threading.Thread(target=data_generator, args=(args, send_data_queue, send_data_lock)).start()
    threading.Thread(target=recv_schedule_thread, args=(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, proc_schedule_list, proc_schedule_lock, _stop_event)).start()
    threading.Thread(target=recv_thread, args=(args.rank, recv_schedule_list, recv_schedule_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event)).start()
    threading.Thread(target=send_thread, args=(args.rank, send_schedule_list, send_schedule_lock, send_data_queue, send_data_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event)).start()

    while _stop_event.is_set() == False:
        inputs, layer_id, p_id, num_outputs = bring_data(recv_data_queue, recv_data_lock, proc_schedule_list, proc_schedule_lock, _stop_event)
        outputs = model(inputs, layer_id)
        print(":::::outputs", outputs.shape, layer_id)
        with send_data_lock:
            send_data_queue.put((p_id, num_outputs, outputs))