import torch
import numpy as np
from torchvision import datasets, transforms 

class Mnist():
    IID = False
    MAX_NUM_CLASSES_PER_CLIENT = 5
    BATCH_SIZE = 100


    def __init__(self):
        # use numpy
        path_np = './mnist.npz'
        f = np.load(path_np)
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
        f.close()



        self.train_data = datasets.MNIST(root='./mnist/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())


        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_data, batch_size=Mnist.BATCH_SIZE, shuffle=True)

    def mnsit_noniid(dataset, num_users):
        """
        Sample non-I.I.D client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return:
        """
        num_shards, num_imgs = 200, 300
        # num_shards is 200, and idx_shard is 0-199
        idx_shard = [i for i in range(num_shards)]
        # print(idx_shard)
        # dict_users is range in num_users  default is 100   type is dictionary
        # 100 ge np array
        dict_users = {i: np.array([]) for i in range(num_users)}
        # print(len(dict_users),dict_users)
        # idxs = 60000, 60000 is devide into 200 parts, and 300 images each part
        # idxs = 1-59999
        idxs = np.arange(num_shards * num_imgs)
        # print(idxs)

        # labels is dataset's label
        # len is 60000
        # it is real label
        labels = dataset.train_labels.numpy()
        # print(len(labels))
        # for i in labels:
        #     print(labels[i])

        # sort labels
        # idxs_labels is the match between 0-59999 and the real labels of dataset
        idxs_labels = np.vstack((idxs, labels))
        # print(idxs_labels)

        # change into arrange depend on the arrange of labels
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        # print(idxs_labels)

        # finally the idxs arrange to labels
        # new ids
        # [30207  5662 55366 ... 23285 15728 11924]
        idxs = idxs_labels[0, :]
        # print(idxs)

        print(idx_shard)
        # divide and assign
        # 100 ge users
        for i in range(num_users):
            # select 2 parts from dataset for every num_users
            # use p a2 = np.random.choice(a=5, size=3, replace=False, p=[0.2, 0.1, 0.3, 0.4, 0.0])
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            # print(rand_set)
            idx_shard = list(set(idx_shard) - rand_set)
            print(idx_shard)
            for rand in rand_set:
                dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        # print(len(dict_users[0]))
        # finally return each user have which 600 images ,every user have a array contain 600 ge number
        return dict_users

    def mnist_iid(dataset, num_users):
        """
        Sample I.I.D. client data from MNIST dataset
        :param dataset:
        :param num_users:
        :return: dict of image index
        """
        num_items = int(len(dataset) / num_users)
        dict_users, all_idxs = {}, [i for i in range(len(dataset))]
        for i in range(num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
        return dict_users

    def get_train_data(self):


        return self.train_loader
    def get_test_data(self):



        return self.test_data