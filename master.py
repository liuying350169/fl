import torch
import torch.distributed.deprecated as dist
import model
import time


def run():
    modell = model.CNN()
    # modell = model.AlexNet()

    size = dist.get_world_size()
    rank = dist.get_rank()

    group_list = []
    for i in range(size):
        group_list.append(i)
    group = dist.new_group(group_list)

    while (1):

        for param in modell.parameters():
            for dst in range(1, size):
                dist.send(param.data, dst=dst)
            # dist.broadcast(param.data, src=0, group=group)

        for param in modell.parameters():
            tensor_update = torch.zeros_like(param.data)
            dist.reduce(tensor_update, dst=0, op=dist.reduce_op.SUM, group=group)


if __name__ == "__main__":
    size = 4
    rank = 0
    dist.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:5000', world_size=size, rank=rank)

    run()
