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
from models.model_encdec import model_encdec
import dataset_invariance
from torch.autograd import Variable
import tqdm

class Trainer():
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the autoencoder.
        :param config: configuration parameters (see train_ae.py)
        """

        self.name_test = str(datetime.datetime.now())[:13]
        self.folder_tensorboard = 'runs/runs-ae/'
        self.folder_test = 'test/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")
        tracks = json.load(open(config.dataset_file))

        self.dim_clip = 180
        print('Creating dataset...')
        self.data_train = dataset_invariance.TrackDataset(tracks,
                                               len_past=config.past_len,
                                               len_future=config.future_len,
                                               train=True,
                                               dim_clip=self.dim_clip
                                               )
        self.train_loader = DataLoader(self.data_train,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True
                                       )
        self.data_test = dataset_invariance.TrackDataset(tracks,
                                              len_past=config.past_len,
                                              len_future=config.future_len,
                                              train=False,
                                              dim_clip=self.dim_clip
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

        # model
        self.mem_n2n = model_encdec(self.settings)
        # loss
        self.criterionLoss = nn.MSELoss()

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
        self.writer = SummaryWriter(self.folder_tensorboard + self.name_test + '_' + config.info)
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

    def draw_track(self, past, future, pred=None, index_tracklet=0, num_epoch=0, train=False):
        """
        Plot past and future trajectory and save it to tensorboard.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param pred: predicted future trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
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

            if (epoch + 1) % 20 == 0:
                print('test on train dataset')
                dict_metrics_train = self.evaluate(self.train_loader, epoch + 1)

                print('test on TEST dataset')
                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)

                # Tensorboard summary: learning rate
                for param_group in self.opt.param_groups:
                    self.writer.add_scalar('learning_rate', param_group["lr"], epoch)

                # Tensorboard summary: train
                self.writer.add_scalar('accuracy_train/eucl_mean', dict_metrics_train['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon10s', dict_metrics_train['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon20s', dict_metrics_train['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon30s', dict_metrics_train['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_train/Horizon40s', dict_metrics_train['horizon40s'], epoch)

                # Tensorboard summary: test
                self.writer.add_scalar('accuracy_test/eucl_mean', dict_metrics_test['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model_ae_epoch_' + str(epoch) + '_' + self.name_test)

                # Tensorboard summary: weights
                for name, param in self.mem_n2n.named_parameters():
                    self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_ae_' + self.name_test)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate the model.
        :param loader: pytorch dataloader to loop over the data
        :param epoch: current epoch (default 0)
        :return: a dictionary with performance metrics
        """

        eucl_mean = horizon10s = horizon20s = horizon30s = horizon40s = 0
        dict_metrics = {}

        # Loop over samples
        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_one_hot) \
                in enumerate(tqdm.tqdm(loader)):
            past = Variable(past)
            future = Variable(future)
            if self.config.cuda:
                past = past.cuda()
                future = future.cuda()
            pred = self.mem_n2n(past, future).data

            distances = torch.norm(pred - future, dim=2)
            eucl_mean += torch.sum(torch.mean(distances, 1))
            horizon10s += torch.sum(distances[:, 9])
            horizon20s += torch.sum(distances[:, 19])
            horizon30s += torch.sum(distances[:, 29])
            horizon40s += torch.sum(distances[:, 39])

            # Draw sample: the first of the batch
            if loader == self.test_loader:
                self.draw_track(past[0],
                                future[0],
                                pred[0],
                                index_tracklet=step,
                                num_epoch=epoch,
                                train=False
                                )

        dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
        dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
        dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
        dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
        dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)

        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_one_hot) \
                in enumerate(tqdm.tqdm(self.train_loader)):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
            self.opt.zero_grad()

            # Get prediction and compute loss
            output = self.mem_n2n(past, future)
            loss = self.criterionLoss(output, future)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()

            # Tensorboard summary: loss
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)

        return loss.item()
