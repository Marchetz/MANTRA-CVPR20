import os
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import datetime
import numpy as np
import cv2
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_controllerMem import model_controllerMem
import dataset_invariance
from torch.autograd import Variable
import io
from PIL import Image
from torchvision.transforms import ToTensor
import time
import tqdm

class Trainer():
    def __init__(self, config):
        """
        The Trainer class handles the training procedure for training the writing controller for the memory.
        :param config: configuration parameters (see train_controllerMem.py)
        """

        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_tensorboard = 'runs/runs-createMem/'
        self.folder_test = 'test/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        tracks = json.load(open(config.track_file))
        self.dim_clip = 180
        print('creating dataset...')
        self.data_train = dataset_invariance.TrackDataset(tracks,
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

        self.data_test = dataset_invariance.TrackDataset(tracks,
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
            "num_prediction": config.preds,
            "past_len": config.past_len,
            "future_len": config.future_len
        }
        self.max_epochs = config.max_epochs
        # load pretrained model and create memory model
        self.model_pretrained = torch.load(config.model_ae)
        self.mem_n2n = model_controllerMem(self.settings, self.model_pretrained)
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len
        
        self.EuclDistance = nn.PairwiseDistance(p=2)
        self.criterionLoss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, 0.5)
        self.iterations = 0
        if config.cuda:
            self.criterionLoss = self.criterionLoss.cuda()
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

        # Tensorboard summary: configuration
        self.writer = SummaryWriter(self.folder_tensorboard + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: {}'.format(self.mem_n2n.name_model), 0)
        self.writer.add_text('Training Configuration', 'dataset train: {}'.format(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: {}'.format(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'batch_size: {}'.format(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: {}'.format(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: {}'.format(self.config.dim_embedding_key), 0)

    def fit(self):
        """
        Writing controller training. The function loops over the data in the training set max_epochs times.
        :return: None
        """
        config = self.config

        # freeze autoencoder layers
        for param in self.mem_n2n.conv_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.conv_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_past.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.encoder_fut.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False

        # Memory Initialization
        self.mem_n2n.init_memory(self.data_train)
        self.save_plot_weight(0)

        # Main training loop
        for epoch in range(self.start_epoch, config.max_epochs):

            self.mem_n2n.init_memory(self.data_train)
            print('epoch: ' + str(epoch))
            start = time.time()
            loss = self._train_single_epoch()
            end = time.time()
            print('Epoch took: {} Loss: {}'.format(end - start, loss))
            self.writer.add_scalar('memory_size/memory_size_train', len(self.mem_n2n.memory_past), epoch)
            self.save_plot_weight(epoch)

            if (epoch + 1) % 20 == 0:
                # Test model while training
                print('start test')
                start_test = time.time()
                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)
                end_test = time.time()
                print('Test took: {}'.format(end_test - start_test))

                # Tensorboard summary: test
                self.writer.add_scalar('accuracy_test/euclMean', dict_metrics_test['eucl_mean'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model_controller_epoch_' + str(epoch) + '_' + self.name_test)

                # print memory on tensorboard
                mem_size = self.mem_n2n.memory_past.shape[0]
                for i in range(mem_size):
                    track_mem = self.mem_n2n.check_memory(i).squeeze(0).cpu().detach().numpy()
                    plt.plot(track_mem[:, 0], track_mem[:, 1], marker='o', markersize=1)
                plt.axis('equal')
                buf = io.BytesIO()
                plt.savefig(buf, format='jpeg')
                buf.seek(0)
                image = Image.open(buf)
                image = ToTensor()(image).unsqueeze(0)
                self.writer.add_image('memory_content/memory', image.squeeze(0), epoch)
                plt.close()

                # save results in a file .txt
                self.save_results(dict_metrics_test, epoch=epoch + 1)

            for name, param in self.mem_n2n.named_parameters():
                self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_controller_' + self.name_test)

    def save_plot_weight(self, epoch):

        fig = plt.figure()
        x = torch.Tensor(np.linspace(0, 1, 100))
        weight = self.mem_n2n.linear_controller.weight.cpu()
        bias = self.mem_n2n.linear_controller.bias.cpu()
        y = torch.sigmoid(weight * x + bias).squeeze()
        plt.plot(x.data.numpy(), y.data.numpy(), '-r', label='y=' + str(weight.item()) + 'x + ' + str(bias.item()))
        plt.plot(x.data.numpy(), [0.5] * 100, '-b')
        plt.title('controller')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper left')
        plt.grid()

        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)

        self.writer.add_image('controller_plot/function', image.squeeze(0), epoch)
        plt.close(fig)

    def save_results(self, dict_metrics_test, epoch=0):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param dict_metrics_train: dictionary with train metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')
        self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')
        self.file.write("TRAIN size: " + str(len(self.data_train)) + '\n')
        self.file.write("TEST size: " + str(len(self.data_test)) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')

        self.file.write("error 1s: " + str(dict_metrics_test['horizon10s'].item()) + '\n')
        self.file.write("error 2s: " + str(dict_metrics_test['horizon20s'].item()) + '\n')
        self.file.write("error 3s: " + str(dict_metrics_test['horizon30s'].item()) + '\n')
        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s'].item()) + '\n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s'].item()) + '\n')
        self.file.write("ADE 2s: " + str(dict_metrics_test['ADE_2s'].item()) + '\n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s'].item()) + '\n')
        self.file.write("ADE 4s: " + str(dict_metrics_test['eucl_mean'].item()) + '\n')

        self.file.close()

    def draw_track(self, past, future, scene_track, pred=None, angle=0, video_id='', vec_id='', index_tracklet=0, num_epoch=0):

        colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.49, 0.33, 0.16), (0.29, 0.57, 0.25)]
        cmap_name = 'scene_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=5)

        fig = plt.figure()
        plt.imshow(scene_track, cmap=cm)
        colors = pl.cm.Reds(np.linspace(1, 0.3, pred.shape[0]))

        matRot_track = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        past = cv2.transform(past.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        future = cv2.transform(future.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        past_scene = past * 2 + self.dim_clip
        future_scene = future * 2 + self.dim_clip
        plt.plot(past_scene[:, 0], past_scene[:, 1], c='blue', linewidth=1, marker='o', markersize=1)
        if pred is not None:
            for i_p in reversed(range(pred.shape[0])):
                pred_i = cv2.transform(pred[i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                pred_scene = pred_i * 2 + self.dim_clip
                plt.plot(pred_scene[:, 0], pred_scene[:, 1], color=colors[i_p], linewidth=0.5, marker='o',
                         markersize=0.5)
        plt.plot(future_scene[:, 0], future_scene[:, 1], c='green', linewidth=1, marker='o', markersize=1)
        plt.title('video: ' + video_id + ', vehicle: ' + vec_id + ',index: ' + str(index_tracklet))
        plt.axis('equal')

        # Save figure in Tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='jpeg')
        buf.seek(0)
        image = Image.open(buf)
        image = ToTensor()(image).unsqueeze(0)
        self.writer.add_image('Image_test/track_' + video_id + '_' + vec_id + '_' + str(index_tracklet).zfill(3),
                              image.squeeze(0), num_epoch)
        plt.close(fig)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """
        self._memory_writing()

        with torch.no_grad():
            dict_metrics = {}
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0

            for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene,
                       scene_one_hot) in enumerate(loader):
                past = Variable(past)
                future = Variable(future)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                pred = self.mem_n2n(past)

                future_rep = future.unsqueeze(1).repeat(1, self.config.preds, 1, 1)
                distances = torch.norm(pred - future_rep, dim=3)
                mean_distances = torch.mean(distances, dim=2)
                index_min = torch.argmin(mean_distances, dim=1)
                min_distances = distances[torch.arange(0, len(index_min)), index_min]

                eucl_mean += torch.sum(torch.mean(min_distances, 1))
                ADE_1s += torch.sum(torch.mean(min_distances[:, :10], 1))
                ADE_2s += torch.sum(torch.mean(min_distances[:, :20], 1))
                ADE_3s += torch.sum(torch.mean(min_distances[:, :30], 1))

                horizon10s += torch.sum(min_distances[:, 9])
                horizon20s += torch.sum(min_distances[:, 19])
                horizon30s += torch.sum(min_distances[:, 29])
                horizon40s += torch.sum(min_distances[:, 39])

                # save in tensorboard: one for each batch
                vid = videos[0]
                vec = vehicles[0]
                num_vec = number_vec[0]
                index_track = index[0].numpy()
                if loader == self.test_loader:
                    self.draw_track(past[0], future[0], scene[0], pred[0], vid, vec + num_vec,
                                    index_tracklet=index_track, num_epoch=epoch)

            dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
            dict_metrics['ADE_1s'] = ADE_1s / len(loader.dataset)
            dict_metrics['ADE_2s'] = ADE_2s / len(loader.dataset)
            dict_metrics['ADE_3s'] = ADE_3s / len(loader.dataset)
            dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
            dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
            dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
            dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)
            self.writer.add_scalar('memory_size/memory_size_test', len(self.mem_n2n.memory_past), epoch)

        return dict_metrics

    def _train_single_epoch(self):
        """
        Training loop over the dataset for an epoch
        :return: loss
        """
        config = self.config
        for step, (index, past, future, _, _, _, _, _, _, _) in enumerate(tqdm.tqdm(self.train_loader)):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
            self.opt.zero_grad()
            prob, sim = self.mem_n2n(past, future)
            # compute loss with best predicted trajectory
            loss = self.CustomLoss(prob, sim)
            loss.backward()
            self.opt.step()
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)

        return loss.item()

    def CustomLoss(self, prob, sim):
        loss = prob * sim + (1 - prob) * (1 - sim)
        return sum(loss)

    def _memory_writing(self):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        self.mem_n2n.init_memory(self.data_train)
        config = self.config
        with torch.no_grad():
            for step, (index, past, future, _, _, _, _, _, _, _) in enumerate(tqdm.tqdm(self.train_loader)):
                self.iterations += 1
                past = Variable(past)
                future = Variable(future)
                if config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                prob, sim = self.mem_n2n(past, future)

    def load(self, directory):
        pass
