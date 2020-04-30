import os
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import datetime
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset_invariance
import index_qualitative
from torch.autograd import Variable
import csv
import time
import tqdm
import pdb


class Validator():
    def __init__(self, config):
        """
        class to evaluate Memnet
        :param config: configuration parameters (see test.py)
        """
        self.index_qualitative = index_qualitative.dict_test
        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'test/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        print('creating dataset...')
        self.dim_clip = 180
        tracks = json.load(open(config.dataset_file))
        self.data_train = dataset_invariance.TrackDataset(tracks,
                                                          len_past=config.past_len,
                                                          len_future=config.future_len,
                                                          train=True,
                                                          dim_clip=self.dim_clip)

        self.train_loader = DataLoader(self.data_train,
                                       batch_size=2,
                                       num_workers=1,
                                       shuffle=True)

        self.data_test = dataset_invariance.TrackDataset(tracks,
                                                         len_past=config.past_len,
                                                         len_future=config.future_len,
                                                         train=False,
                                                         dim_clip=self.dim_clip)

        self.test_loader = DataLoader(self.data_test,
                                      batch_size=config.batch_size,
                                      num_workers=1,
                                      shuffle=False)

        print('dataset created')
        if config.visualize_dataset:
            print('save examples in folder test')
            self.data_train.save_dataset(self.folder_test + 'dataset_train/')
            self.data_test.save_dataset(self.folder_test + 'dataset_test/')
            print('Saving complete!')

        # load model to evaluate
        self.mem_n2n = torch.load(config.model)
        self.mem_n2n.num_prediction = config.preds
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len

        self.EuclDistance = nn.PairwiseDistance(p=2)
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

    def test_model(self):
        """
        Memory selection and evaluation!
        :return: None
        """
        # populate the memory
        start = time.time()
        self._memory_writing(self.config.saved_memory)
        end = time.time()
        print('writing time: ' + str(end-start))

        # run test!
        dict_metrics_test = self.evaluate(self.test_loader)
        self.save_results(dict_metrics_test)

    def save_results(self, dict_metrics_test):
        """
        Serialize results
        :param dict_metrics_test: dictionary with test metrics
        :param dict_metrics_train: dictionary with train metrics
        :param epoch: epoch index (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')

        self.file.write("model:" + self.config.model + '\n')
        self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')
        self.file.write("TRAIN size: " + str(len(self.data_train)) + '\n')
        self.file.write("TEST size: " + str(len(self.data_test)) + '\n')

        self.file.write("error 1s: " + str(dict_metrics_test['horizon10s']) + 'm \n')
        self.file.write("error 2s: " + str(dict_metrics_test['horizon20s']) + 'm \n')
        self.file.write("error 3s: " + str(dict_metrics_test['horizon30s']) + 'm \n')
        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s']) + 'm \n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s']) + 'm \n')
        self.file.write("ADE 2s: " + str(dict_metrics_test['ADE_2s']) + 'm \n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s']) + 'm \n')
        self.file.write("ADE 4s: " + str(dict_metrics_test['eucl_mean']) + 'm \n')

        self.file.close()

    def draw_track(self, past, future, scene_track, pred=None, angle=0, video_id='', vec_id='', index_tracklet=0,
                    path='', horizon_dist=None):
        """
        Plot past and future trajectory and save it to test folder.
        :param past: the observed trajectory
        :param future: ground truth future trajectory
        :param pred: predicted future trajectory
        :param angle: rotation angle to plot the trajectory in the original direction
        :param video_id: video index of the trajectory
        :param vec_id: vehicle type of the trajectory
        :param pred: predicted future trajectory
        :param: the observed scene where is the trajectory
        :param index_tracklet: index of the trajectory in the dataset (default 0)
        :param num_epoch: current epoch (default 0)
        :return: None
        """

        colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.49, 0.33, 0.16), (0.29, 0.57, 0.25)]
        cmap_name = 'scene_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
        fig = plt.figure()
        plt.imshow(scene_track, cmap=cm)
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
                plt.plot(pred_scene[:, 0], pred_scene[:, 1], color=colors[i_p], linewidth=0.5, marker='o', markersize=0.5)
        plt.plot(future_scene[:, 0], future_scene[:, 1], c='green', linewidth=1, marker='o', markersize=1)
        plt.title('FDE 1s: ' + str(horizon_dist[0]) + ' FDE 2s: ' + str(horizon_dist[1]) + ' FDE 3s: ' +
                  str(horizon_dist[2]) + ' FDE 4s: ' + str(horizon_dist[3]))
        plt.axis('equal')
        plt.savefig(path + video_id + '_' + vec_id + '_' + str(index_tracklet).zfill(3) + '.png')
        plt.close(fig)

    def evaluate(self, loader):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :return: dictionary of performance metrics
        """

        self.mem_n2n.eval()
        with torch.no_grad():
            dict_metrics = {}
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0

            for step, (index, past, future, presents, angle_presents, videos, vehicles, number_vec, scene, scene_one_hot) \
                    in enumerate(tqdm.tqdm(loader)):
                past = Variable(past)
                future = Variable(future)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                if self.config.withIRM:
                    scene_one_hot = Variable(scene_one_hot)
                    scene_one_hot = scene_one_hot.cuda()
                    pred = self.mem_n2n(past, scene_one_hot)
                else:
                    pred = self.mem_n2n(past)

                future_rep = future.unsqueeze(1).repeat(1, self.config.preds, 1, 1)
                distances = torch.norm(pred - future_rep, dim=3)
                mean_distances = torch.mean(distances, dim=2)
                index_min = torch.argmin(mean_distances, dim=1)
                distance_pred = distances[torch.arange(0, len(index_min)), index_min]

                horizon10s += sum(distance_pred[:, 9])
                horizon20s += sum(distance_pred[:, 19])
                horizon30s += sum(distance_pred[:, 29])
                horizon40s += sum(distance_pred[:, 39])
                ADE_1s += sum(torch.mean(distance_pred[:, :10], dim=1))
                ADE_2s += sum(torch.mean(distance_pred[:, :20], dim=1))
                ADE_3s += sum(torch.mean(distance_pred[:, :30], dim=1))
                eucl_mean += sum(torch.mean(distance_pred[:, :40], dim=1))

                if self.config.saveImages is not None:
                    for i in range(len(past)):
                        horizon_dist = [round(distance_pred[i, 9].item(), 3), round(distance_pred[i, 19].item(), 3),
                                        round(distance_pred[i, 29].item(), 3), round(distance_pred[i, 39].item(), 3)]
                        vid = videos[i]
                        vec = vehicles[i]
                        num_vec = number_vec[i]
                        index_track = index[i].numpy()
                        angle = angle_presents[i].cpu()

                        if self.config.saveImages == 'All':
                            if not os.path.exists(self.folder_test + vid):
                                os.makedirs(self.folder_test + vid)
                            video_path = self.folder_test + vid + '/'
                            if not os.path.exists(video_path + vec + num_vec):
                                os.makedirs(video_path + vec + num_vec)
                            vehicle_path = video_path + vec + num_vec + '/'
                            self.draw_track(past[i], future[i], scene[i], pred[i], angle, vid, vec + num_vec,
                                            index_tracklet=index_track, path=vehicle_path, horizon_dist=horizon_dist)
                        if self.config.saveImages == 'Subset':
                            if index_track.item() in self.index_qualitative[vid][vec + num_vec]:
                                # Save interesting results
                                if not os.path.exists(self.folder_test + 'highlights'):
                                    os.makedirs(self.folder_test + 'highlights')
                                highlights_path = self.folder_test + 'highlights' + '/'
                                self.draw_track(past[i], future[i], scene[i], pred[i], angle, vid, vec + num_vec,
                                                index_tracklet=index_track, path=highlights_path, horizon_dist=horizon_dist)

            dict_metrics['eucl_mean'] = round((eucl_mean / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_1s'] = round((ADE_1s / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_2s'] = round((ADE_2s / len(loader.dataset)).item(), 3)
            dict_metrics['ADE_3s'] = round((ADE_3s / len(loader.dataset)).item(), 3)
            dict_metrics['horizon10s'] = round((horizon10s / len(loader.dataset)).item(), 3)
            dict_metrics['horizon20s'] = round((horizon20s / len(loader.dataset)).item(), 3)
            dict_metrics['horizon30s'] = round((horizon30s / len(loader.dataset)).item(), 3)
            dict_metrics['horizon40s'] = round((horizon40s / len(loader.dataset)).item(), 3)

        return dict_metrics

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """

        if saved_memory:
            self.mem_n2n.memory_past = torch.load(self.config.memories_path + 'memory_past.pt')
            self.mem_n2n.memory_fut = torch.load(self.config.memories_path + 'memory_fut.pt')
        else:
            self.mem_n2n.init_memory(self.data_train)
            config = self.config
            with torch.no_grad():
                for step, (index, past, future, _, _, _, _, _, _, scene_one_hot) in enumerate(tqdm.tqdm(self.train_loader)):
                    past = Variable(past)
                    future = Variable(future)
                    if config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                    if self.config.withIRM:
                        scene_one_hot = Variable(scene_one_hot)
                        scene_one_hot = scene_one_hot.cuda()
                        self.mem_n2n.write_in_memory(past, future, scene_one_hot)
                    else:
                        self.mem_n2n.write_in_memory(past, future)

                # save memory
                torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
                torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')
