import threading, time, argparse, os, pickle, numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

SCHEDULE_TAG = 0
QUEUE_LENGTH = 10
SEND_FLAG = 0
RECV_FLAG = 1
schedule_shape = ['recv/send flag', 'layer_id', 'src', 'dst', 'input_height', 'input_width', 'input_channel', 'slicing_start', 'slicing_end', 'tag', 'proc_flag']

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

def recv_thread(rank, recv_schedule_list, recv_schedule_lock, recv_data_list, recv_data_lock, internal_data_list, internal_data_lock, _stop_event):
    while _stop_event.is_set() == False:
        # 현재 proc을 위해 필요한 데이터를 받아옴
        if len(recv_schedule_list) > 0:
            data = []
            threads = []
            # 스레드를 열고 input data를 동시에 받음. 이를 위해서 input에 해당하는 schedule을 모두 pop해서 recv함.
            while True:
                with recv_schedule_lock:
                    schedule = recv_schedule_list.pop(0)
                data.append(torch.empty(size=(1, schedule[6], schedule[5], schedule[8]-schedule[7]+1)))
                if schedule[2].item() == rank: # recv/irecv는 자기자신에게 보낼경우 segfault남.
                    while True:
                        if schedule[9].item() in internal_data_list:
                            with internal_data_lock:
                                data[-1] = internal_data_list.pop(schedule[9].item())
                        else:
                            time.sleep(0.000001)
                else:
                    threads.append(dist.irecv(tensor=data[-1], src=schedule[2].item(), tag=schedule[9].item()))
                if len(recv_schedule_list) < 1:
                    time.sleep(0.000001)
                elif recv_schedule_list[0][10] == True:
                    break
            for recv in threads:
                recv.wait()
            # scheduling decision에 있는 애들이 모두 받아졌으면 merge함
            input_data = torch.cat(data)
            print("aaaa", input_data.shape, [d.shape for d in data])
            with recv_data_lock:
                recv_data_list.append(input_data)
        else:
            time.sleep(0.000001)

def send_thread(rank, send_schedule_list, send_schedule_lock, send_data_list, send_data_lock, internal_data_list, internal_data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(send_schedule_list) > 0 and len(send_data_list) > 0:
            with send_schedule_lock:
                schedule = send_schedule_list.pop(0)
            with send_data_lock:
                output_data = send_data_list.pop(0)
            data = torch.index_select(input=output_data, dim=3, index=torch.IntTensor([schedule[7], schedule[8]]))
            if schedule[3].item() == rank: # send/isend는 자기자신에게 보낼경우 segfault남.
                with internal_data_lock:
                    internal_data_list[schedule[9].item()] = data
            else:
                dist.isend(tensor=data, dst=schedule[3].item(), tag=schedule[9].item())
        else:
            time.sleep(0.000001)

# smart cameras
def send_request():
    request = torch.empty(len(schedule_shape), dtype=torch.int32)
    dist.send(tensor=request, dst=0, tag=SCHEDULE_TAG)

def recv_schedule_thread(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, proc_schedule_list, proc_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        schedule = torch.empty(len(schedule_shape), dtype=torch.int32)
        dist.recv(tensor=schedule, src=0, tag=SCHEDULE_TAG)
        if schedule[0] == RECV_FLAG:
            if schedule[10] == True:
                with proc_schedule_lock:
                    proc_schedule_list.append(schedule[1])
            with recv_schedule_lock:
                recv_schedule_list.append(schedule)
            # print(recv_schedule_list, proc_schedule_list)
        elif schedule[0] == SEND_FLAG:
            with send_schedule_lock:
                send_schedule_list.append(schedule)
        # print("schedule queue length", len(recv_schedule_list), len(send_schedule_list))

# edge server
def recv_request():
    request = torch.empty(len(schedule_shape), dtype=torch.int32)
    return dist.recv(tensor=request, src=None, tag=SCHEDULE_TAG)

def send_schedule(schedule, dst):
    dist.send(tensor=schedule, dst=dst, tag=SCHEDULE_TAG)

def processing(model, inputs, proc_schedule_list, proc_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(proc_schedule_list) > 0:
            with proc_schedule_lock:
                i = proc_schedule_list.pop(0)
            print("process", inputs.shape)
            return model(inputs, i)
        else:
            time.sleep(0.000001)