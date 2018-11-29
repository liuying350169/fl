import numpy as np
import keras
import random
import tensorflow as tf
from keras.datasets import mnist
from keras import backend as K

class DataSource(object):
    def __init__(self):
        raise NotImplementedError()
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        raise NotImplementedError()
    def sample_single_non_iid(self, weight=None):
        raise NotImplementedError()

# NUM_CLASS_BIG = 2000
# NUM_CLASS_LITTLE = 100
# BASE_SELECT = 0
# CLASS_WEIGHT = [0.42,0.42,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02]
#CLASS_WEIGHT = [0,0,0,0,0,0,0.5,0.5,0,0]

# NUM_CLASS_BIG = 2000
# NUM_CLASS_LITTLE = 1000
# BASE_SELECT = 0
# CLASS_WEIGHT = [0.02,0.02,0.02,0.02,0.02,0.02,0.42,0.42,0.02,0.02]

NUM_CLASS_BIG = 2000
NUM_CLASS_LITTLE = 1000
BASE_SELECT = 300
CLASS_WEIGHT = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]


class Mnist(DataSource):

    ##IID = False
    IID = True
    MAX_NUM_CLASSES_PER_CLIENT = 10
    
    def __init__(self):

        
        
        #use numpy
        path_np = './mnist.npz'
        f = np.load(path_np)
        x_train,y_train = f['x_train'],f['y_train']
        x_test,y_test = f['x_test'],f['y_test']
        f.close()        
        #1 & 2
        
        x0 = x_train[y_train == 0][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y0 = y_train[y_train == 0][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x1 = x_train[y_train == 1][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y1 = y_train[y_train == 1][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x2 = x_train[y_train == 2][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y2 = y_train[y_train == 2][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT] 
        x3 = x_train[y_train == 3][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y3 = y_train[y_train == 3][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x4 = x_train[y_train == 4][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y4 = y_train[y_train == 4][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x5 = x_train[y_train == 5][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y5 = y_train[y_train == 5][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x6 = x_train[y_train == 6][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y6 = y_train[y_train == 6][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x7 = x_train[y_train == 7][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y7 = y_train[y_train == 7][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x8 = x_train[y_train == 8][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y8 = y_train[y_train == 8][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        x9 = x_train[y_train == 9][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        y9 = y_train[y_train == 9][BASE_SELECT:NUM_CLASS_LITTLE+BASE_SELECT]
        
        #x_test = x_test[y_test == 1]
        #y_test = y_test[y_test == 1]

        self.x = np.concatenate([x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]).astype('float')
        self.y = np.concatenate([y0,y1,y2,y3,y4,y5,y6,y7,y8,y9])
        
#         self.x = x_train[0:1300]
#         self.y = y_train[0:1300]
#         x_test = x_test[0:300]
#         y_test = y_test[0:300]        
#         self.x = np.concatenate([x_train,x_test]).astype('float')
#         self.y = np.concatenate([y_train,y_test])        
        

        n = self.x.shape[0]        
        idx = np.arange(n)
        np.random.shuffle(idx)
        #change: delete random op
        self.x = self.x[idx]  # n * 28 * 28
        self.y = self.y[idx]  # n * 1        
        
        data_split = (0.6, 0.3, 0.1)
        num_train = int(n * data_split[0])
        num_test = int(n * data_split[1])
        self.x_train = self.x[0:num_train]
        self.x_test = self.x[num_train:num_train + num_test]
        self.x_valid = self.x[num_train + num_test:]
        self.y_train = self.y[0:num_train]
        self.y_test = self.y[num_train:num_train + num_test]
        self.y_valid = self.y[num_train + num_test:]
    
        self.classes = np.unique(self.y)
        print("Mnist __init__")
    
    def gen_dummy_non_iid_weights(self):
        self.classes = np.array(range(10))
        num_classes_this_client = random.randint(1, Mnist.MAX_NUM_CLASSES_PER_CLIENT + 1)
        classes_this_client = random.sample(self.classes.tolist(), num_classes_this_client)
        w = np.array([random.random() for _ in range(num_classes_this_client)])
        weights = np.array([0.] * self.classes.shape[0])
        for i in range(len(classes_this_client)):
            weights[classes_this_client[i]] = w[i]
        weights /= np.sum(weights)
        print("Mnist gen_dummy_non_iid_weights")
        return weights.tolist()
            


    # assuming client server already agreed on data format
    def post_process(self, xi, yi):
        if K.image_data_format() == 'channels_first':
            xi = xi.reshape(1, xi.shape[0], xi.shape[1])
        else:
            xi = xi.reshape(xi.shape[0], xi.shape[1], 1)

        y_vec = keras.utils.to_categorical(yi, self.classes.shape[0])
        #print("Mnist post_processs")-----------------------------------------------------------------
        return xi / 255., y_vec
    

    # split evenly into exact num_workers chunks, with test_reserve globally
    def partitioned_by_rows(self, num_workers, test_reserve=.3):
        n_test = int(self.x.shape[0] * test_reserve)
        n_train = self.x.shape[0] - n_test
        nums = [n_train // num_workers] * num_workers
        nums[-1] += n_train % num_workers
        idxs = np.array([np.random.choice(np.arange(n_train), num, replace=False) for num in nums])
        print("Mnist partitioned_by_rows")
        return {
            # (size_partition * 28 * 28, size_partition * 1) * num_partitions
            "train": [post_process(self.x[idx], self.y[idx]) for idx in idxs],
            # (n_test * 28 * 28, n_test * 1)
            "test": post_process(self.x[np.arange(n_train, n_train + n_test)], self.y[np.arange(n_train, n_train + n_test)])
        }

    # Generate one sample from all available data, *with replacement*.
    # This is to simulate date generation on a client.
    # weight: [probablity of classes]
    # returns: 28 * 28, 1
    def sample_single_non_iid(self, x, y, weight=None):
        # first pick class, then pick a datapoint at random
        chosen_class = np.random.choice(self.classes,p=CLASS_WEIGHT)
        #print(chosen_class)
        candidates_idx = np.array([i for i in range(y.shape[0]) if y[i] == chosen_class])
        idx = np.random.choice(candidates_idx)
        #print("Mnist sample_single_non_iid")----------------------------------------------------------
        return self.post_process(x[idx], y[idx])

    
    # generate t, t, v dataset given distribution and split
    def fake_non_iid_data(self, min_train=100, max_train=1000, data_split=(.6,.3,.1)):        
        # my_class_distr = np.array([np.random.random() for _ in range(self.classes.shape[0])])
        # my_class_distr /= np.sum(my_class_distr)
        my_class_distr = [1. / self.classes.shape[0] * self.classes.shape[0]] if Mnist.IID \
                else self.gen_dummy_non_iid_weights()
        
        #train_size = random.randint(min_train, max_train)
        train_size = max_train
        test_size = int(train_size / data_split[0] * data_split[1])
        valid_size = int(train_size / data_split[0] * data_split[2])

        train_set = [self.sample_single_non_iid(self.x_train, self.y_train, my_class_distr) for _ in range(train_size)]
        test_set = [self.sample_single_non_iid(self.x_test, self.y_test, my_class_distr) for _ in range(test_size)]
        valid_set = [self.sample_single_non_iid(self.x_valid, self.y_valid, my_class_distr) for _ in range(valid_size)]
#         train_set = []
#         test_set = []
#         valid_set = []
        print("done generating fake data")

        return ((train_set, test_set, valid_set), my_class_distr)


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        m = Mnist()
        # res = m.partitioned_by_rows(9)
        # print(res["test"][1].shape)
        print("Mnist __main__")
        #m.fake_non_iid_data()
        for _ in range(10):
            print(m.gen_dummy_non_iid_weights())



