from common import *
import cv2, logging


def data_generator(data_list, data_lock):
    # video data load
    vid = cv2.VideoCapture(args.data_path+args.video_name)
    fps = vid.get(cv2.CAP_PROP_FPS)
    delay = int(600/fps)
    roi_mask = cv2.imread(args.data_path+args.roi_name, cv2.IMREAD_UNCHANGED)
    roi_mask = cv2.resize(roi_mask, args.resolution, interpolation=cv2.INTER_CUBIC)

    kernel = None
    backgroundObject = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=128, detectShadows=False)
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
        info = torch.Tensor(boxedFrame.shape).type(dtype=torch.int16)
        send_schedule(info)
        with data_lock:
            data_list.append(boxedFrame)

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
    parser.add_argument('--data_path', default='./Data/', type=str, help='Image frame data path')
    parser.add_argument('--video_name', default='vdo.avi', type=str, help='Video file name')
    parser.add_argument('--roi_name', default='roi.jpg', type=str, help='RoI file name')
    parser.add_argument('--num_proc', default=2, type=int, help='Number of processes')
    parser.add_argument('--resolution', default=(854, 480), type=tuple, help='Image resolution')
    parser.add_argument('--verbose', default=False, type=str2bool, help='If you want to print debug messages, set True')
    args = parser.parse_args()

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

    threading.Thread(target=schedule_recv_thread, args=(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event)).start()
    threading.Thread(target=recv_thread, args=(recv_schedule_list, recv_schedule_lock, recv_data_list, recv_data_lock, _stop_event)).start()
    threading.Thread(target=send_thread, args=(send_schedule_list, send_schedule_lock, send_data_list, send_data_lock, _stop_event)).start()
    threading.Thread(target=data_generator, args=(send_data_list, send_data_lock)).start()

    while _stop_event.is_set() == False:
        inputs = bring_data(recv_data_list, recv_data_lock, _stop_event)
        outputs = processing(inputs)
        with send_data_lock:
            send_data_list.append(outputs)