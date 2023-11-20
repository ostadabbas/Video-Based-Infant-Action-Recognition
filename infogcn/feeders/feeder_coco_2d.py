import numpy as np
import pickle
import json
import random
import math
import glob
import os
import os.path as osp

from torch.utils.data import Dataset

class Feeder(Dataset):
    def __init__(self, data_path, split=None, p_interval=None, repeat=1, random_choose=False, random_shift=False, random_move=False, random_rot=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True, bone=False, vel=False, sort=False, **kwargs):


        self.data_path = data_path
        if 'ntu' in self.data_path:
            train_split_name = 'xsub_train'
            test_split_name = 'xsub_val'
        else:
            train_split_name = 'train'
            test_split_name = 'val'

        with open(data_path, 'rb') as f:
            data_file = pickle.load(f)
        file_split = data_file['split']
        annotations = data_file['annotations']
        if 'test' in split:
            self.train_val = 'test'
            self.data_dict = [item for item in annotations if item['frame_dir'] in file_split[test_split_name]]
        else:
            self.train_val = 'train'
            self.data_dict = [item for item in annotations if item['frame_dir'] in file_split[train_split_name]]


        self.time_steps = 100
        self.bone = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 5),
                     (12, 6), (9, 7), (7, 5), (10, 8), (8, 6), (5, 0),
                     (6, 0), (1, 0), (3, 1), (2, 0), (4, 2)]
        
        self.data = []
        for item in self.data_dict:
            kps = item['keypoint'][0]
            aux_z = np.ones_like(kps)[...,0]
            kps = np.dstack([kps, aux_z])
            self.data.append(kps)
        self.label = [item['label'] for item in self.data_dict]

        self.debug = debug
        self.data_path = data_path
        self.label_path = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.repeat = repeat
        if normalization:
            self.get_mean_map()


    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        if self.train_val=='train':
            return len(self.data_dict)*self.repeat
        else:
            return len(self.data_dict)

    def __iter__(self):
        return self

    def rand_view_transform(self,X, agx, agy, s):
        agx = math.radians(agx)
        agy = math.radians(agy)
        Rx = np.asarray([[1,0,0], [0,math.cos(agx),math.sin(agx)], [0, -math.sin(agx),math.cos(agx)]])
        Ry = np.asarray([[math.cos(agy), 0, -math.sin(agy)], [0,1,0], [math.sin(agy), 0, math.cos(agy)]])
        Ss = np.asarray([[s,0,0],[0,s,0],[0,0,s]])
        X0 = np.dot(np.reshape(X,(-1,3)), np.dot(Ry,np.dot(Rx,Ss)))
        X = np.reshape(X0, X.shape)
        return X

    def __getitem__(self, index):
        label = self.label[index % len(self.data_dict)]
        value = self.data[index % len(self.data_dict)]

        if self.train_val == 'train':
            random.random()
            agx = random.randint(-60, 60)
            agy = random.randint(-60, 60)
            s = random.uniform(0.5, 1.5)

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0) + 1e-6)
            scalerValue = scalerValue*2-1
            scalerValue = np.reshape(scalerValue, (-1, 17, 3))

            data = np.zeros( (self.time_steps, 17, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            random_idx = random.sample(list(np.arange(length))*100, self.time_steps)
            random_idx.sort()
            data[:,:,:] = value[random_idx,:,:]
            data[:,:,:] = value[random_idx,:,:]

        else:
            random.random()
            agx = 0
            agy = 0
            s = 1.0

            center = value[0,1,:]
            value = value - center
            scalerValue = self.rand_view_transform(value, agx, agy, s)

            scalerValue = np.reshape(scalerValue, (-1, 3))
            scalerValue = (scalerValue - np.min(scalerValue,axis=0)) / (np.max(scalerValue,axis=0) - np.min(scalerValue,axis=0) + 1e-6)
            scalerValue = scalerValue*2-1

            scalerValue = np.reshape(scalerValue, (-1, 17, 3))

            data = np.zeros( (self.time_steps, 17, 3) )

            value = scalerValue[:,:,:]
            length = value.shape[0]

            idx = np.linspace(0,length-1,self.time_steps).astype(int)
            data[:,:,:] = value[idx,:,:] # T,V,C
            
        data = np.transpose(data, (2, 0, 1))
        C,T,V = data.shape
        data = np.reshape(data,(C,T,V,1))

        return data, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()

        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
