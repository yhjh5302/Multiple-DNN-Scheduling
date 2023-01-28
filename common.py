import threading, time, argparse, os, copy, pickle, queue, numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms

SCHEDULE_TAG = 0
schedule_shape = ['layer_id', 'num_inputs', 'src', 'dst', 'input_height', 'input_width', 'input_channel', 'slicing_start', 'slicing_end', 'tag', 'proc_flag']

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def bring_data(recv_data_queue, recv_data_lock, proc_schedule_list, proc_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(proc_schedule_list) > 0:
            with proc_schedule_lock:
                schedule = proc_schedule_list.pop(0)
            layer_id = schedule[0]
            num_partition = schedule[1]
            start_tag = schedule[2]
            data_list = []
            time.sleep(1)
            print("(bring data) num_partition", num_partition, layer_id, start_tag)
            while recv_data_queue.qsize() < num_partition:
                time.sleep(0.000001) # wait for data recv
            for i in range(num_partition):
                with recv_data_lock:
                    tag, data, job = recv_data_queue.get()
                    print("(bring data) tag", tag)
                if tag != start_tag+i:
                    print("(bring data) tag err", tag, start_tag+i)
                data_list.append(data)
                print("(bring data)", tag, "wait")
                if job != None:
                    job.wait()
            print(torch.cat(data_list).shape)
            print("Processed!", layer_id)
            return torch.cat(data_list), layer_id
        else:
            time.sleep(0.000001) # wait for data recv

def recv_thread(rank, recv_schedule_list, recv_schedule_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(recv_schedule_list) > 0:
            with recv_schedule_lock:
                schedule = recv_schedule_list.pop(0)
            src = schedule[2].item()
            tag = schedule[9].item()
            input_channel = schedule[6]
            input_width = schedule[5]
            input_height = schedule[8] - schedule[7]
            data = torch.empty(size=(1, input_channel, input_width, input_height))
            if src == rank: # recv/irecv는 자기자신에게 보낼경우 segfault남.
                while len(internal_data_list) == 0:
                    time.sleep(0.000001) # wait for data recv
                with internal_data_lock:
                    data_tag, data = internal_data_list.pop(0)
                if data_tag != tag:
                    print("(recv_thread) tag err", data_tag, tag)
                with recv_data_lock:
                    recv_data_queue.put([tag, data, None])
                    print("(recv_thread) ", tag, data.shape, None)
            else:
                with recv_data_lock:
                    job = dist.irecv(tensor=data, src=src, tag=tag)
                    recv_data_queue.put([tag, data, job])
                    print("(recv_thread) ", tag, data.shape, job)
            print("recv_thread recv_data_lock done")
        else:
            time.sleep(0.000001)

def send_thread(rank, send_schedule_list, send_schedule_lock, send_data_list, send_data_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(send_data_list) > 0:
            with send_data_lock:
                layer_id, data = send_data_list.pop(0)
            while True:
                if len(send_schedule_list) > 0:
                    with send_schedule_lock:
                        schedule = send_schedule_list.pop(0)
                    if schedule[0] != layer_id:
                        break
                    dst = schedule[3].item()
                    tag = schedule[9].item()
                    slicing_index = (schedule[7].item(), schedule[8].item())
                    data = output_data[:,:,:,slicing_index[0]:slicing_index[1]+1].contiguous()
                    print("(send_thread) ", data.shape, tag, dst)
                    if dst == rank: # send/isend는 자기자신에게 보낼경우 segfault남.
                        with internal_data_lock:
                            internal_data_list.append((tag, data))
                            print("(send_thread) ", tag, data.shape)
                    else:
                        dist.isend(tensor=data, dst=dst, tag=tag)
                else:
                    time.sleep(0.000001) # wait for data recv
        else:
            time.sleep(0.000001) # wait for data recv

# smart cameras
def send_request():
    request = torch.empty(len(schedule_shape), dtype=torch.int32)
    dist.send(tensor=request, dst=0, tag=SCHEDULE_TAG)

def recv_schedule_thread(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, proc_schedule_list, proc_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        schedule = torch.empty(len(schedule_shape), dtype=torch.int32)
        dist.recv(tensor=schedule, src=0, tag=SCHEDULE_TAG)
        if schedule[2] >= 0:
            if schedule[10] == True:
                with proc_schedule_lock:
                    proc_schedule_list.append((schedule[0].item(), schedule[1].item(), schedule[9].item()))
            with recv_schedule_lock:
                recv_schedule_list.append(schedule)
        elif schedule[3] >= 0:
            with send_schedule_lock:
                send_schedule_list.append(schedule)
        # print("schedule queue length", len(recv_schedule_list), len(send_schedule_list))

# edge server
def recv_request():
    request = torch.empty(len(schedule_shape), dtype=torch.int32)
    return dist.recv(tensor=request, src=None, tag=SCHEDULE_TAG)

def send_schedule(schedule, dst):
    dist.send(tensor=schedule, dst=dst, tag=SCHEDULE_TAG)