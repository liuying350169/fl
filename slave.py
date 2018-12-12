import torch
import torch.distributed.deprecated as dist
from datasource import Mnist
import model
import time
import copy
from torch.multiprocessing import Process

MAX_EPOCH = 50
LR = 0.001


def get_new_model(model):
    for param in model.parameters():
        dist.recv(param.data, src=0)
    # print(dist.get_rank())
    return model


def run(size, rank):
    modell = model.CNN()
    # modell = model.AlexNet()

    optimizer = torch.optim.Adam(modell.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    # size = dist.get_world_size()
    # rank = dist.get_rank()

    train_loader = Mnist().get_train_data()

    test_data = Mnist().get_test_data()
    test_x = torch.unsqueeze(test_data.test_data, dim=1).type(
        torch.FloatTensor) / 255.  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_data.test_labels

    group_list = []
    for i in range(size):
        group_list.append(i)
    group = dist.new_group(group_list)

    for epoch in range(MAX_EPOCH):

        modell = get_new_model(modell)
        # current_model = copy.deepcopy(modell)

        for step, (b_x, b_y) in enumerate(train_loader):
            # modell = get_new_model(modell)
            # current_model = copy.deepcopy(modell)

            output = modell(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # new_model = copy.deepcopy(modell)

        # for param1, param2 in zip( current_model.parameters(), new_model.parameters() ):
        # dist.reduce(param2.data-param1.data, dst=0, op=dist.reduce_op.SUM, group=group)

        for param in modell.parameters():
            dist.reduce(param, dst=0, op=dist.reduce_op.SUM, group=group)

        test_output, last_layer = modell(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        print('Epoch: ', epoch, ' Rank: ', rank, '| train loss: %.4f' % loss.data.numpy(),
              '| test accuracy: %.2f' % accuracy)


def init_processes(size, rank, run):
    dist.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:5000', world_size=size, rank=rank)
    run(size, rank)


if __name__ == "__main__":
    size = 4
    processes = []
    for rank in range(1, size):
        p = Process(target=init_processes, args=(size, rank, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()