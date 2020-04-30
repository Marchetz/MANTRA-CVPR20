import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import datetime
import cv2
from random import randint
import numpy as np
import json
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models.model_memory_IRM import model_memory_IRM
import io
from PIL import Image
from torchvision.transforms import ToTensor
import dataset_invariance
import index_qualitative
import tqdm


class Trainer:
    def __init__(self, config):
        """
        Trainer class for training the Iterative Refinement Module (IRM)
        :param config: configuration parameters (see train_IRM.py)
        """

        self.index_qualitative = index_qualitative.dict_test
        self.name_run = 'runs/runs-IRM/'
        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'training/training_IRM/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")
        tracks = json.load(open(config.dataset_file))

        self.dim_clip = 180
        print('creating dataset...')
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
        print('dataset created')

        self.num_prediction = config.preds
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "num_prediction": self.num_prediction,
            "past_len": config.past_len,
            "future_len": config.future_len
        }
        self.max_epochs = config.max_epochs

        # load pretrained model and create memory_model
        self.model = torch.load(config.model)
        self.mem_n2n = model_memory_IRM(self.settings, self.model)
        self.mem_n2n.past_len = config.past_len
        self.mem_n2n.future_len = config.future_len

        self.criterionLoss = nn.MSELoss()
        self.opt = torch.optim.Adam(self.mem_n2n.parameters(), lr=config.learning_rate)
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
        self.writer = SummaryWriter(self.name_run + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: ' + self.mem_n2n.name_model, 0)
        self.writer.add_text('Training Configuration', 'dataset train: ' + str(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: ' + str(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'number of prediction: ' + str(self.num_prediction), 0)
        self.writer.add_text('Training Configuration', 'batch_size: ' + str(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: ' + str(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: ' + str(self.settings["dim_embedding_key"]),
                             0)

    def write_details(self):
        """
        Serialize configuration parameters to file.
        """
        self.file.write("points of past track: " + str(self.config.past_len) + '\n')
        self.file.write("points of future track: " + str(self.config.future_len) + '\n')
        self.file.write("train size: " + str(len(self.data_train)) + '\n')
        self.file.write("test size: " + str(len(self.data_test)) + '\n')
        self.file.write("batch size: " + str(self.config.batch_size) + '\n')

    def fit(self):
        """
        Iterative refinement model training. The function loops over the data in the training set max_epochs times.
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
        for param in self.mem_n2n.linear_controller.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.convScene_1.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.convScene_2.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.RNN_scene.parameters():
            param.requires_grad = True
        for param in self.mem_n2n.fc_refine.parameters():
            param.requires_grad = True

        # Load memory
        # populate the memory
        start = time.time()
        self._memory_writing(self.config.saved_memory)
        self.writer.add_text('Training Configuration', 'memory size: ' + str(len(self.mem_n2n.memory_past)), 0)
        end = time.time()
        print('writing time: ' + str(end-start))

        step_results = [1, 10, 20, 30, 40, 50, 60, 80, 90, 100, 120, 150, 170, 200, 250, 300, 350, 400, 450, 490, 550, 600]
        # Main training loop
        for epoch in range(self.start_epoch, config.max_epochs):
            self.mem_n2n.train()

            print('epoch: ' + str(epoch))
            start = time.time()
            loss = self._train_single_epoch()
            end = time.time()
            print('Epoch took: {} Loss: {}'.format(end - start, loss))

            if (epoch + 1) in step_results:
                # Test model while training
                print('start test')
                start_test = time.time()
                dict_metrics_test = self.evaluate(self.test_loader, epoch + 1)
                end_test = time.time()
                print('Test took: {}'.format(end_test - start_test))

                # Tensorboard summary: test
                self.writer.add_scalar('accuracy_test/euclMean', dict_metrics_test['euclMean'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)
                self.writer.add_scalar('dimension_memory/memory', len(self.mem_n2n.memory_past), epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model_IRM_epoch_' + str(epoch) + '_' + self.name_test)
                self.save_results(dict_metrics_test, epoch=epoch + 1)

            for name, param in self.mem_n2n.named_parameters():
                self.writer.add_histogram(name, param.data, epoch)

        # Save final trained model
        torch.save(self.mem_n2n, self.folder_test + 'model_mantra_' + self.name_test)

    def save_results(self, dict_metrics_test, epoch=0):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')
        self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')

        self.file.write("error 1s: " + str(dict_metrics_test['horizon10s'].item()) + '\n')
        self.file.write("error 2s: " + str(dict_metrics_test['horizon20s'].item()) + '\n')
        self.file.write("error 3s: " + str(dict_metrics_test['horizon30s'].item()) + '\n')
        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s'].item()) + '\n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s'].item()) + '\n')
        self.file.write("ADE 2s: " + str(dict_metrics_test['ADE_2s'].item()) + '\n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s'].item()) + '\n')
        self.file.write("ADE 4s: " + str(dict_metrics_test['euclMean'].item()) + '\n')

        self.file.close()

    def draw_track(self, past, future, scene, pred=None, angle=0, video_id='', vec_id='', index_tracklet=0,
                   num_epoch=0):
        """
        Plot past and future trajectory and save it to tensorboard.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param scene: the observed scene where is the trajectory
        :param pred: predicted future trajectory
        :param angle: rotation angle to plot the trajectory in the original direction
        :param video_id: video index of the trajectory
        :param vec_id: vehicle type of the trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :return: None
        """

        colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.49, 0.33, 0.16), (0.29, 0.57, 0.25)]
        cmap_name = 'scene_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
        fig = plt.figure()
        plt.imshow(scene, cmap=cm)
        colors = pl.cm.Reds(np.linspace(1, 0.3, pred.shape[0]))
        matRot_track = cv2.getRotationMatrix2D((0, 0), -angle, 1)
        past = cv2.transform(past.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        future = cv2.transform(future.cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
        story_scene = past * 2 + self.dim_clip
        future_scene = future * 2 + self.dim_clip
        plt.plot(story_scene[:, 0], story_scene[:, 1], c='blue', linewidth=1, marker='o', markersize=1)
        if pred is not None:
            for i_p in reversed(range(pred.shape[0])):
                pred_i = cv2.transform(pred[i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                pred_scene = pred_i * 2 + self.dim_clip
                plt.plot(pred_scene[:, 0], pred_scene[:, 1], color=colors[i_p], linewidth=0.5, marker='o',
                         markersize=0.5)
        plt.plot(future_scene[:, 0], future_scene[:, 1], c='green', linewidth=1, marker='o', markersize=1)
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

        self.mem_n2n.eval()
        with torch.no_grad():
            dict_metrics = {}
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0

            for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene,
                       scene_one_hot) in enumerate(tqdm.tqdm(loader)):
                past = Variable(past)
                future = Variable(future)
                scene_one_hot = Variable(scene_one_hot)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    scene_one_hot = scene_one_hot.cuda()
                pred = self.mem_n2n(past, scene_one_hot)

                future_rep = future.unsqueeze(1).repeat(1, self.num_prediction, 1, 1)
                distances = torch.norm(pred - future_rep, dim=3)
                distances_mean = torch.mean(distances, dim=2)
                index_min = torch.argmin(distances_mean, dim=1)
                distance_pred = distances[torch.arange(past.shape[0]), index_min[:]]
                horizon10s += sum(distance_pred[:, 9])
                horizon20s += sum(distance_pred[:, 19])
                horizon30s += sum(distance_pred[:, 29])
                horizon40s += sum(distance_pred[:, 39])
                ADE_1s += sum(torch.mean(distance_pred[:, :10], dim=1))
                ADE_2s += sum(torch.mean(distance_pred[:, :20], dim=1))
                ADE_3s += sum(torch.mean(distance_pred[:, :30], dim=1))
                eucl_mean += sum(torch.mean(distance_pred[:, :40], dim=1))

                for i in range(len(past)):
                    vid = videos[i]
                    vec = vehicles[i]
                    num_vec = number_vec[i]
                    index_track = index[i].numpy()
                    angle = angle_presents[i].cpu()
                    if loader == self.test_loader and self.config.saveImages:
                        if index_track.item() in self.index_qualitative[vid][vec + num_vec]:
                            self.draw_track(past[i], future[i], scene[i], pred[i], angle, vid, vec + num_vec,
                                            index_tracklet=index_track, num_epoch=epoch)

            dict_metrics['euclMean'] = eucl_mean / len(loader.dataset)
            dict_metrics['ADE_1s'] = ADE_1s / len(loader.dataset)
            dict_metrics['ADE_2s'] = ADE_2s / len(loader.dataset)
            dict_metrics['ADE_3s'] = ADE_3s / len(loader.dataset)
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
        self.mem_n2n.train()
        for step, (index, past, future, _, _, _, _, _, _, scene_one_hot) in enumerate(tqdm.tqdm(self.train_loader)):
            self.iterations += 1
            past = Variable(past)
            future = Variable(future)
            scene_one_hot = Variable(scene_one_hot)
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
                scene_one_hot = scene_one_hot.cuda()
            self.opt.zero_grad()
            output = self.mem_n2n(past, scene_one_hot)

            future_repeat = future.unsqueeze(1).repeat(1, self.num_prediction, 1, 1)
            distances = torch.norm(output - future_repeat, dim=3)
            distances_mean = torch.mean(distances, dim=2)
            index_min = torch.argmin(distances_mean, dim=1)
            best_pred = output[torch.arange(past.shape[0]), index_min[:]]
            loss = self.criterionLoss(best_pred, future)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mem_n2n.parameters(), 1.0, norm_type=2)
            self.opt.step()
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)

        return loss.item()

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """

        if saved_memory:
            print('memories of pretrained model')
        else:
            self.mem_n2n.init_memory(self.data_train)
            config = self.config
            with torch.no_grad():
                for step, (index, past, future, _, _, _, _, _, _, _) in enumerate(self.train_loader):
                    self.iterations += 1
                    past = Variable(past)
                    future = Variable(future)
                    if config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                    self.mem_n2n.write_in_memory(past, future)

        # save memory
        torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
        torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')

