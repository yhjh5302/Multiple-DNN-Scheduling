import threading, time, argparse, os, pickle, queue, numpy as np
import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing as mp

SCHEDULE_TAG = 0
SCHEDULE_TAG_2 = 2100000000
num_pieces = 3
num_partitions = 27
num_classes = 80
schedule_shape = ['layer_id', 'num_inputs', 'num_outputs', 'pred_id', 'p_id', 'src', 'dst', 'input_height', 'input_width', 'input_channel',
                  'slicing_start', 'slicing_end', 'tag', 'proc_flag']

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
                proc_schedule = proc_schedule_list.pop(0)
            layer_id = proc_schedule[0]
            num_inputs = proc_schedule[1].item()
            num_outputs = proc_schedule[2].item()
            p_id = proc_schedule[4].item()
            start_tag = proc_schedule[12].item()
            data_list = []
            # print("(bring data) num_inputs", num_inputs, layer_id, start_tag)
            while recv_data_queue.qsize() < num_inputs:
                time.sleep(0.001) # wait for data recv
            for i in range(num_inputs):
                with recv_data_lock:
                    tag, data, job = recv_data_queue.get()
                #     print("(bring data) tag", tag)
                # if tag != start_tag+i:
                #     print("(bring data) tag err", tag, start_tag+i)
                data_list.append(data)
                # print("(bring data)", tag, "wait")
                if job != None:
                    job.join()
            # print([d.shape for d in data_list])
            # print(torch.cat(data_list, dim=-1).shape)
            return torch.cat(data_list, dim=-1), layer_id, p_id, num_outputs
        else:
            time.sleep(0.001) # wait for data recv

def recv_thread(rank, recv_schedule_list, recv_schedule_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(recv_schedule_list) > 0:
            with recv_schedule_lock:
                schedule = recv_schedule_list.pop(0)
            src = schedule[5].item()
            tag = schedule[12].item()
            input_channel = schedule[9]
            input_width = schedule[8]
            input_height = schedule[11] - schedule[10] + 1
            data = torch.empty(size=(1, input_channel, input_width, input_height))
            if src == rank: # recv/irecv는 자기자신에게 보낼경우 segfault남.
                while len(internal_data_list) == 0:
                    time.sleep(0.001) # wait for data recv
                with internal_data_lock:
                    data_tag, data = internal_data_list.pop(0)
                # if data_tag != tag:
                #     print("(recv_thread) tag err", data_tag, tag)
                with recv_data_lock:
                    recv_data_queue.put([tag, data, None])
                    # print("(recv_thread) ", tag, data.shape, None)
            else:
                with recv_data_lock:
                    job = threading.Thread(target=dist.recv, kwargs={"tensor":data, "src":src, "tag":tag})
                    job.start()
                    recv_data_queue.put([tag, data, job])
            #         print("(recv_thread) ", tag, data.shape)
            # print("recv_thread recv_data_lock done")
        else:
            time.sleep(0.001)

def send_thread(rank, send_schedule_list, send_schedule_lock, send_data_list, send_data_lock, recv_data_queue, recv_data_lock, internal_data_list, internal_data_lock, _stop_event):
    while _stop_event.is_set() == False:
        if len(send_data_list) > 0:
            with send_data_lock:
                pred_id, num_outputs, outputs = send_data_list.pop(0)
            for i in range(num_outputs):
                idx = None
                while True:
                    for i, s in enumerate(send_schedule_list):
                        if s[3].item() == pred_id:
                            idx = i
                            # print("(send_thread) send", s)
                            break
                    if idx != None:
                        break
                    # else:
                    #     print("(send_thread) waiting", len(send_schedule_list), pred_id, outputs.shape)
                    #     time.sleep(5)
                # send_schedule중에 pred_id가 동일한거만 꺼냄
                with send_schedule_lock:
                    schedule = send_schedule_list.pop(idx)
                dst = schedule[6].item()
                tag = schedule[12].item()
                slicing_index = (schedule[10].item(), schedule[11].item() + 1)
                if outputs.dim() == 4:
                    data = outputs[:,:,:,slicing_index[0]:slicing_index[1]].contiguous()
                elif outputs.dim() == 2:
                    data = outputs[:,slicing_index[0]:slicing_index[1]].contiguous()
                # print("(send_thread) ", data.shape, tag, dst, len(send_data_list))
                if dst == rank: # send/isend는 자기자신에게 보낼경우 segfault남.
                    with internal_data_lock:
                        internal_data_list.append((tag, data))
                        # print("(send_thread) ", tag, data.shape)
                else:
                    threading.Thread(target=dist.send, kwargs={"tensor":data, "dst":dst, "tag":tag}).start()
        else:
            time.sleep(0.001) # wait for data recv

def recv_schedule_thread(recv_schedule_list, recv_schedule_lock, send_schedule_list, send_schedule_lock, proc_schedule_list, proc_schedule_lock, _stop_event):
    while _stop_event.is_set() == False:
        schedule = torch.empty(len(schedule_shape), dtype=torch.int32)
        dist.recv(tensor=schedule, src=0, tag=SCHEDULE_TAG)
        if schedule[5] >= 0:
            if schedule[13] == True:
                with proc_schedule_lock:
                    proc_schedule_list.append(schedule)
            with recv_schedule_lock:
                recv_schedule_list.append(schedule)
        elif schedule[6] >= 0:
            with send_schedule_lock:
                send_schedule_list.append(schedule)
        # print("schedule queue length", len(recv_schedule_list), len(send_schedule_list))

def send_schedule(schedule, dst):
    dist.send(tensor=schedule, dst=dst, tag=SCHEDULE_TAG)

# smart cameras
def send_request():
    request = torch.empty(len(schedule_shape), dtype=torch.int32)
    dist.send(tensor=request, dst=0, tag=SCHEDULE_TAG)
    p_tag = torch.empty(2, dtype=torch.int32)
    dist.recv(tensor=p_tag, src=0, tag=SCHEDULE_TAG_2)
    print("p_tag", p_tag[0].item())
    return p_tag[0].item()

# edge server
def recv_request(p_tag):
    request = torch.empty(len(schedule_shape), dtype=torch.int32)
    src = dist.recv(tensor=request, src=None, tag=SCHEDULE_TAG)
    tag_tensor = torch.empty(2, dtype=torch.int32)
    tag_tensor[0] = p_tag
    dist.send(tensor=tag_tensor, dst=src, tag=SCHEDULE_TAG_2)
    print("p_tag", tag_tensor)
    return src