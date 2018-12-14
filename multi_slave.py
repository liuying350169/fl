import torch
import torch.distributed.deprecated as dist
from datasource import Mnist, Mnist_noniid
import model
import time
import copy
from torch.multiprocessing import Process

MAX_EPOCH = 500
LR = 0.001
IID = False

def get_new_model(model, group):
    for param in model.parameters():
        dist.broadcast(param.data, src=0, group=group)
    #print(dist.get_rank())
    return  model

def run(size, rank):


    modell = model.CNN()
    #modell = model.AlexNet()

    optimizer = torch.optim.Adam(modell.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()



    if(IID == True):
        train_loader = Mnist().get_train_data()
        test_data = Mnist().get_test_data()
    else:
        if(rank > 0):
            if(rank == 1):
                train_loader = Mnist_noniid().get_train_data1()
                test_data = Mnist_noniid().get_test_data1()
            if(rank == 2):
                train_loader = Mnist_noniid().get_train_data2()
                test_data = Mnist_noniid().get_test_data2()
            if(rank == 3):
                train_loader = Mnist_noniid().get_train_data3()
                test_data = Mnist_noniid().get_test_data3()
            if(rank == 4):
                train_loader = Mnist_noniid().get_train_data4()
                test_data = Mnist_noniid().get_test_data4()
            if(rank == 5):
                train_loader = Mnist_noniid().get_train_data5()
                test_data = Mnist_noniid().get_test_data5()

    #size = dist.get_world_size()
    #rank = dist.get_rank()

    #train_loader = Mnist().get_train_data()
    #test_data = Mnist().get_test_data()

    for step, (b_x, b_y) in enumerate(test_data):
        test_x = b_x
        test_y = b_y

    group_list = []
    for i in range(size):
        group_list.append(i)
    group = dist.new_group(group_list)

    for epoch in range(MAX_EPOCH):

        modell = get_new_model(modell, group)
        #current_model = copy.deepcopy(modell)

        test_output, last_layer = modell(test_x)
        pred_y = torch.max(test_output, 1)[1].data.numpy()
        accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
        # print_str = 'Epoch: ', epoch, ' Rank: ', rank, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy


        for step, (b_x, b_y) in enumerate(train_loader):

            #modell = get_new_model(modell)
            #current_model = copy.deepcopy(modell)

            output = modell(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()   
            optimizer.step()

        for param in modell.parameters():
            dist.reduce(param.data, dst=0, op=dist.reduce_op.SUM, group=group)

        f = open('./test.txt', 'a')
        print('Epoch: ', epoch, ' Rank: ', rank, '| train loss: %.4f' % loss.data.numpy(),
              '| test accuracy: %.2f' % accuracy, file=f)
        print('Epoch: ', epoch, ' Rank: ', rank, '| train loss: %.4f' % loss.data.numpy(),
              '| test accuracy: %.2f' % accuracy)
        #f.writelines(print_str)
        #f.write('\n Epoch: ', epoch, ' Rank: ', rank, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)


def init_processes(size, rank, run):
    dist.init_process_group(backend='tcp', init_method='tcp://127.0.0.1:5000', world_size=size, rank=rank)
    run(size, rank)

if __name__ == "__main__":
    size = 6
    processes = []
    for rank in range(1, size):
        p = Process(target=init_processes, args=(size, rank, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
