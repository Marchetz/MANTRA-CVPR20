import os
import matplotlib.pyplot as plt
import datetime
import cv2
from random import randint
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.model_memory_single import model_memory_single
import dataset
from torch.autograd import Variable


class MemoryWriter():
    def __init__(self, config):
        """
        Class for writing samples in memory
        :param config: configuration parameters (see train_save_memory.py)
        """

        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'test/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        tracks = json.load(open(config.track_file))

        self.dim_clip = 180
        print('creating dataset...')
        self.data_train = dataset.TrackDataset(tracks,
                                               num_instances=config.past_len,
                                               num_labels=config.future_len,
                                               train=True,
                                               dim_clip=self.dim_clip
                                               )

        self.train_loader = DataLoader(self.data_train,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True
                                       )

        self.train_loader_mem = DataLoader(self.data_train,
                                           batch_size=config.batch_size,
                                           num_workers=1,
                                           shuffle=False
                                           )

        self.data_test = dataset.TrackDataset(tracks,
                                              num_instances=config.past_len,
                                              num_labels=config.future_len,
                                              train=False,
                                              dim_clip=self.dim_clip
                                              )

        self.test_loader = DataLoader(self.data_test,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False
                                      )
        print('dataset created')

        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": config.preds
        }

        # load pretrained model and create memory model
        self.model_pretrained = torch.load(config.model)
        self.mem_n2n = model_memory_single(self.settings, self.model_pretrained)
        self.EuclDistance = nn.PairwiseDistance(p=2)
        self.iterations = 0
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

    def mem_write(self):
        """
        Loop over samples and populate memory
        :return: None
        """
        config = self.config

        print('Updating memory')
        for step, (_, past, future, _, _, _, _, _, scene) in enumerate(self.train_loader_mem):
            self.iterations += 1
            if config.rotation_aug:
                for i_rotate in range(len(past)):
                    angle = randint(0, 360)
                    rot = cv2.getRotationMatrix2D((0, 0), angle, 1)
                    past[i_rotate, :, :2] = torch.FloatTensor(cv2.transform(past[i_rotate, :, :2].numpy().reshape(-1, 1, 2), rot).squeeze())
                    future[i_rotate, :, :2] = torch.FloatTensor(cv2.transform(future[i_rotate, :, :2].numpy().reshape(-1, 1, 2), rot).squeeze())
            past = Variable(past)
            future = Variable(future)
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
            self.mem_n2n.create_memory(past, future)

        # save memory
        torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
        torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')

    def load(self, directory):
        pass
