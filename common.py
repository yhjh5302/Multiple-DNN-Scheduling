import threading, time, argparse, os, pickle
import torch
import torch.distributed as dist

SCHEDULE_TAG = 0
QUEUE_LENGTH = 10

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def bring_data(data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(data_list) > 0:
            with data_lock:
                data = data_list.pop(0)
            return data
        else:
            time.sleep(0.000001) # wait for data download

def recv_thread(schedule_list, schedule_lock, data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        # 스케줄 리스트에서 받아야할 데이터가 있으면
        if len(schedule_list) > 0:
            with schedule_lock:
                schedules = schedule_list.pop(0)
            # 스레드를 열고 input data를 동시에 받음
            data = []
            for i, (src, input_shape, tag) in enumerate(schedules):
                data.append(torch.Empty(input_shape))
                threading.Thread(target=dist.recv, kwargs={'tensor':data[i], 'src':src, 'tag':tag}).start()
            # scheduling decision에 있는 애들이 모두 받아졌으면 merge함
            input_data = torch.cat(data)
            with data_lock:
                data_list.append(input_data)
        else:
            time.sleep(0.000001)

def send_thread(schedule_list, schedule_lock, data_list, data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(schedule_list) > 0 and len(data_list) > 0:
            with schedule_lock:
                schedules = schedule_list.pop(0)
            # output data를 받아 schedule대로 조각내고 목적지로 전송
            with data_lock:
                output_data = data_list.pop(0)
            for (dst, slice_shape, tag) in schedules:
                data = torch.index_select(input=output_data, dim=2, index=torch.IntTensor(slice_shape))
                threading.Thread(target=dist.send, kwargs={'tensor':data, 'dst':dst, 'tag':tag}).start()
        else:
            time.sleep(0.000001)

def send_schedule(info):
    dist.send(tensor=info, dst=0, tag=SCHEDULE_TAG)

def schedule_recv_thread(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        schedule_list = torch.empty(schedule_shape, dtype=torch.int16)
        dist.recv(tensor=schedule_list, src=0, tag=SCHEDULE_TAG)
        with recv_schedule_lock:
            recv_schedule_list.extend(schedule_list[0])
        with send_schedule_lock:
            send_schedule_list.extend(schedule_list[1])

def processing(inputs):
    return 0