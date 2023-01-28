import torch, queue
recv_data_queue = queue.PriorityQueue()
tag = 1
data = [tag, torch.empty(size=(4, 3, 2, 1)), None]
print("data", data[0], "queue len", recv_data_queue.qsize())
recv_data_queue.put(data)
recv_data_queue.put([8, torch.empty(size=(4, 3, 2, 5)), None])
recv_data_queue.put([3, torch.empty(size=(1, 3, 2, 2)), None])
recv_data_queue.put([4, torch.empty(size=(1, 3, 2, 2)), None])
print("data", data[0], "queue len", recv_data_queue.qsize())
del data
print("queue len", recv_data_queue.qsize())
print("data", recv_data_queue.get()[0])
print("data", recv_data_queue.get()[0])
print("data", recv_data_queue.get()[0])
print("data", recv_data_queue.get()[0])