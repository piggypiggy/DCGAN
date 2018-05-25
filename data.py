import os
import numpy as np 
import struct

class DataProvider(object):
    def __init__(self, shuffle=True):
        train_data_path = 'train-images-idx3-ubyte'
        label_data_path = 'train-labels-idx1-ubyte'
        self.train_data = self.read_data(train_data_path)
        self.label_data = self.read_data(label_data_path)
        if shuffle:
            perm = np.arange(self.train_data.shape[0])
            np.random.shuffle(perm) # train和label用相同的置换
            self.train_data = self.train_data[perm]
            self.label_data = self.label_data[perm]
        self.cursor = 0
        self.train_num = len(self.train_data) 

    def read_data(self, path):
        if (not os.path.exists(path)):
            print('file not exists')
            return

        with open(path, 'rb') as f:
            magic = struct.unpack('>I', f.read(4))[0]
            if (magic == 2051): # train
                num, height, width = struct.unpack('>III', f.read(12))
                data = np.fromstring(f.read(), dtype=np.uint8).reshape(num, height, width, 1).astype(np.float_)
                data = (data / 255.0 * 2 - 1)
                return data
            elif (magic == 2049): # label
                num = struct.unpack('>I', f.read(4))[0]
                data1 = np.fromstring(f.read(), dtype=np.uint8).reshape(num, 1)
                data = np.zeros((num, 10), dtype=np.float)
                for i in range(num):
                    data[i, data1[i]] = 1.0
                return data
        return None

    def next_batch(self, batch_size):
        if self.cursor + batch_size > self.train_num:
            perm = np.arange(self.train_num)
            np.random.shuffle(perm) # train和label用相同的置换
            self.train_data = self.train_data[perm]
            self.label_data = self.label_data[perm]
            self.cursor = 0
        next_train_batch = self.train_data[self.cursor:self.cursor + batch_size]
        next_label_batch = self.label_data[self.cursor:self.cursor + batch_size]
        self.cursor += batch_size
        return next_train_batch, next_label_batch

    def get_train_num(self):
        return self.train_num

if __name__ == '__main__':
    d = DataProvider()
    print(d.label_data[0:20])



