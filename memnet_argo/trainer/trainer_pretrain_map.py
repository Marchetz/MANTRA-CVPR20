import os
import matplotlib.pyplot as plt
import datetime
import io
from PIL import Image
from torchvision.transforms import ToTensor
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_ae_map import model_ae_map
from models.model_ae_map_v2 import model_ae_map_v2
import numpy as np

from dataset import dataset_argo_map
from torch.autograd import Variable
import pdb
import time
import h5py
import pickle



class Trainer():
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'test/' + self.name_test
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")

        print('Creating dataset...')

        self.data_train = torch.load('data/data_train_map.pt')
        self.data_test = torch.load('data/data_val_map.pt')

        self.train_loader = DataLoader(self.data_train,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True
                                       )

        self.test_loader = DataLoader(self.data_test,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False
                                      )
        print('Dataset created')

        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_feature_tracklet": config.past_len * 2,
            "dim_feature_future": config.future_len * 2,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": config.future_len,

        }
        self.max_epochs = config.max_epochs

        self.mem_n2n = model_ae_map_v2(self.settings)
        # loss
        self.criterionLoss = nn.MSELoss()
        self.criterionLoss_map = nn.BCELoss()
        self.criterionLoss_map2 = nn.CrossEntropyLoss()

        self.EuclDistance = nn.PairwiseDistance(p=2)
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.5)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Write details to file
        self.write_details()
        self.file.close()

        # Tensorboard summary: configuration
        self.writer = SummaryWriter('runs-pretrain/' + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: {}'.format(self.mem_n2n.name_model), 0)
        self.writer.add_text('Training Configuration', 'dataset train: {}'.format(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: {}'.format(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'batch_size: {}'.format(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: {}'.format(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: {}'.format(self.config.dim_embedding_key), 0)

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """

        self.file.write('points of past track: {}'.format(self.config.past_len) + '\n')
        self.file.write('points of future track: {}'.format(self.config.future_len) + '\n')
        self.file.write('train size: {}'.format(len(self.data_train)) + '\n')
        self.file.write('test size: {}'.format(len(self.data_test)) + '\n')
        self.file.write('batch size: {}'.format(self.config.batch_size) + '\n')
        self.file.write('embedding dim: {}'.format(self.config.dim_embedding_key) + '\n')

    def draw_track(self, past, future, pred=None, index_tracklet=0, num_epoch=0, save_fig=False, train=False):
        """
        Plot past and future trajectory and save it to tensorboard.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param pred: predicted future trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :param save_fig: True or False if you want to save the plot as figure to Tensorboard
        :param train: True or False, indicates whether the sample is in the training or testing set
        :return: None
        """

        fig = plt.figure()
        past = past.cpu().numpy()
        future = future.cpu().numpy()
        plt.plot(past[:, 0], past[:, 1], c='blue', marker='o', markersize=3)
        plt.plot(future[:, 0], future[:, 1], c='green', marker='o', markersize=3)

        if pred is not None:
            pred = pred.cpu().numpy()
            plt.plot(pred[:, 0], pred[:, 1], color='red', linewidth=1, marker='o', markersize=1)
        plt.axis('equal')

        if save_fig:
            # Save figure in Tensorboard
            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            image = Image.open(buf)
            image = ToTensor()(image).unsqueeze(0)

            if train:
                self.writer.add_image('Image_train/track' + str(index_tracklet), image.squeeze(0), num_epoch)
            else:
                self.writer.add_image('Image_test/track' + str(index_tracklet), image.squeeze(0), num_epoch)

        plt.close(fig)

    def fit(self):
        """
        Autoencoder training procedure. The function loops over the data in the training set max_epochs times.
        :return: None
        """

        config = self.config
        # Training loop
        for epoch in range(self.start_epoch, config.max_epochs):
            print(' ----- Epoch: {}'.format(epoch))
            loss = self._train_single_epoch()
            print('Loss: {}'.format(loss))

            step_results = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 65, 70, 90, 100]
            if (epoch + 1) in step_results:
                print('test')
                #dict_metrics_train = self.evaluate(self.train_loader, epoch + 1)
                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)

                # Tensorboard summary: learning rate
                for param_group in self.opt.param_groups:
                    self.writer.add_scalar('learning_rate', param_group["lr"], epoch)

                # Tensorboard summary: train
                # self.writer.add_scalar('accuracy_train/euclMean', dict_metrics_train['euclMean'], epoch)
                # #self.writer.add_scalar('accuracy_train/Horizon01s', dict_metrics_train['horizon01s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon10s', dict_metrics_train['horizon10s'], epoch)
                # #self.writer.add_scalar('accuracy_train/Horizon20s', dict_metrics_train['horizon20s'], epoch)
                # self.writer.add_scalar('accuracy_train/Horizon30s', dict_metrics_train['horizon30s'], epoch)
                # #self.writer.add_scalar('accuracy_train/Horizon40s', dict_metrics_train['horizon40s'], epoch)

                # Tensorboard summary: test
                self.writer.add_scalar('accuracy_test/euclMean', dict_metrics_test['euclMean'], epoch)
                #self.writer.add_scalar('accuracy_test/Horizon01s', dict_metrics_test['horizon01s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                #self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                #self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model_epoch_' + str(epoch) + '_' + self.name_test)

                # Tensorboard summary: weights
                for name, param in self.mem_n2n.named_parameters():
                    self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_ae' + self.name_test)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """
        count_plot_image = 0
        euclidean_mean = 0
        horizon10s = 0
        horizon30s = 0
        dict_metrics = {}
        list_error = []
        config = self.config
        # Loop over samples
        self.mem_n2n.eval()
        with torch.no_grad():
            for step, (index, past, future, presents, angle_presents, scene) in enumerate(loader):
                past = Variable(past)
                future = Variable(future)
                scene_input = Variable(torch.Tensor(np.eye(2, dtype=np.float32)[scene]).permute(0, 3, 1, 2))
                scene_gt = scene.type(torch.LongTensor)
                if config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    scene_input = scene_input.cuda()
                pred, pred_map = self.mem_n2n(past, future, scene_input)
                pred = pred.data

                for i in range(len(past)):
                    dist = self.EuclDistance(pred[i], future[i, :, :2])
                    euclidean_mean += torch.mean(dist)
                    list_error.append(torch.mean(dist))
                    horizon10s += dist[9]
                    horizon30s += dist[29]

                    # Draw sample
                    if loader == self.test_loader and i == 0 and count_plot_image < 100:
                        self.draw_track(past[i],
                                        future[i],
                                        pred[i],
                                        index_tracklet=index[i],
                                        num_epoch=epoch,
                                        save_fig=True,
                                        train=False
                                        )
                        count_plot_image += 1
                    # if loader == self.train_loader and self.max_epochs and i == 0:
                    #     self.draw_track(past[i],
                    #                     future[i],
                    #                     pred[i],
                    #                     index_tracklet=step,
                    #                     num_epoch=epoch,
                    #                     save_fig=True,
                    #                     train=True
                    #                     )

            dict_metrics['euclMean'] = euclidean_mean / len(loader.dataset)
            dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
            dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)


        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        self.mem_n2n.train()
        for step, (index, past, future, presents, angle_presents, scene) in enumerate(self.train_loader):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            scene_input = Variable(torch.Tensor(np.eye(2, dtype=np.float32)[scene]).permute(0,3,1,2))
            scene_gt = scene.type(torch.cuda.LongTensor)
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
                scene_input = scene_input.cuda()
            self.opt.zero_grad()

            # Get prediction and compute loss
            #try:
            pred, map_pred = self.mem_n2n(past, future, scene_input)
            #map_pred = torch.permute(0,2,3,1).contiguous().squeeze(3)

            loss_pred = self.criterionLoss(pred, future)
            loss_map = self.criterionLoss_map2(map_pred, scene_gt)


            loss = loss_pred + loss_map
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

            # Tensorboard summary: loss
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)
            self.writer.add_scalar('loss/loss_pred', loss_pred, self.iterations)
            self.writer.add_scalar('loss/loss_map', loss_map, self.iterations)

        return loss.item()
