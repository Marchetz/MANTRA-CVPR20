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
from models.model_decoder import model_decoder


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
        tracks = json.load(open(config.dataset_file))

        self.dim_clip = 180
        print('creating dataset...')
        self.data_train = dataset_invariance.TrackDataset(tracks,
                                                          train=True,
                                                          dim_clip=self.dim_clip)

        self.train_loader = DataLoader(self.data_train,
                                       batch_size=2,
                                       num_workers=1,
                                       shuffle=True)

        self.data_test = dataset_invariance.TrackDataset(tracks,
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

        #load model to evaluate
        self.mem_n2n = torch.load(config.model)
        self.mem_n2n.num_prediction = config.preds
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len

        self.EuclDistance = nn.PairwiseDistance(p=2)
        self.iterations = 0
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

    def test_model(self):
        """
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

        self.file.write("error 1s: " + str(dict_metrics_test['horizon10s']) + '\n')
        self.file.write("error 2s: " + str(dict_metrics_test['horizon20s']) + '\n')
        self.file.write("error 3s: " + str(dict_metrics_test['horizon30s']) + '\n')
        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s']) + '\n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s']) + '\n')
        self.file.write("ADE 2s: " + str(dict_metrics_test['ADE_2s']) + '\n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s']) + '\n')
        self.file.write("ADE 4s: " + str(dict_metrics_test['eucl_mean']) + '\n')

        self.file.close()

    def draw_track(self, past, future, scene_track, pred=None, angle=0, video_id='', vec_id='', index_tracklet=0,
                   save_fig=False, path='', horizon_dist=None):

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

        if save_fig:
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
                                            index_tracklet=index_track, save_fig=True,
                                            path=vehicle_path, horizon_dist=horizon_dist)
                        if self.config.saveImages == 'Subset':
                            if index_track.item() in self.index_qualitative[vid][vec + num_vec]:
                                # Save interesting results
                                if not os.path.exists(self.folder_test + 'highlights'):
                                    os.makedirs(self.folder_test + 'highlights')
                                highlights_path = self.folder_test + 'highlights' + '/'
                                self.draw_track(past[i], future[i], scene[i], pred[i], angle, vid, vec + num_vec,
                                                index_tracklet=index_track, save_fig=True,
                                                path=highlights_path, horizon_dist=horizon_dist)

            dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
            dict_metrics['ADE_1s'] = ADE_1s / len(loader.dataset)
            dict_metrics['ADE_2s'] = ADE_2s / len(loader.dataset)
            dict_metrics['ADE_3s'] = ADE_3s / len(loader.dataset)
            dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
            dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
            dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
            dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)

        return dict_metrics

    def _memory_writing(self, saved_memory):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """

        if saved_memory:
            # self.mem_n2n.memory_past = torch.load(self.config.memories_path + 'memory_past.pt')
            # self.mem_n2n.memory_fut = torch.load(self.config.memories_path + 'memory_fut.pt')
            print('ok memory')
        else:
            self.mem_n2n.init_memory(self.data_train)
            config = self.config
            with torch.no_grad():
                for step, (index, past, future, _, _, _, _, _, _, scene_one_hot) in enumerate(tqdm.tqdm(self.train_loader)):
                    self.iterations += 1
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

    def evaluate_slim(self, loader):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """

        self.mem_n2n.eval()

        videos_json = {}
        test_ids = self.data_test.ids_split_test
        for ids in test_ids:
            videos_json['video_' + str(ids).zfill(4)] = {}
            for i_temp in range(self.data_test.video_length[str(ids).zfill(4)]):
                videos_json['video_' + str(ids).zfill(4)]['frame_' + str(i_temp).zfill(3)] = {}

        with torch.no_grad():
            dict_metrics = {}
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon10s = horizon20s = horizon30s = horizon40s = 0
            list_err1 = []
            list_err2 = []
            list_err3 = []
            list_err4 = []

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

                for i in range(len(past)):
                    list_error_single_pred_mean = []
                    list_error_single_pred_1s = []
                    list_error_single_pred_2s = []
                    list_error_single_pred_3s = []
                    list_error_single_pred_4s = []
                    for i_multiple in range(len(pred[i])):
                        pred_one = pred[i][i_multiple]
                        dist = self.EuclDistance(pred_one, future[i, :])
                        list_error_single_pred_mean.append(round(torch.mean(dist).item(),3))
                        list_error_single_pred_1s.append(round(dist[9].item(), 3))
                        list_error_single_pred_2s.append(round(dist[19].item(), 3))
                        list_error_single_pred_3s.append(round(dist[29].item(), 3))
                        list_error_single_pred_4s.append(round(dist[39].item(), 3))
                    # i_min = np.argmin(list_error_single_pred_mean)
                    i_min = np.argmin(list_error_single_pred_4s)


                    dist = self.EuclDistance(pred[i][i_min], future[i, :])
                    he_1s = round(dist[9].item(), 3)
                    he_2s = round(dist[19].item(), 3)
                    he_3s = round(dist[29].item(), 3)
                    he_4s = round(dist[39].item(), 3)
                    eucl_mean += round(torch.mean(dist).item(), 3)
                    ADE_1s += round(torch.mean(dist[:10]).item(), 3)
                    ADE_2s += round(torch.mean(dist[:20]).item(), 3)
                    ADE_3s += round(torch.mean(dist[:30]).item(), 3)
                    list_err1.append(he_1s)
                    list_err2.append(he_2s)
                    list_err3.append(he_3s)
                    list_err4.append(he_4s)
                    horizon10s += he_1s
                    horizon20s += he_2s
                    horizon30s += he_3s
                    horizon40s += he_4s

            dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
            dict_metrics['ADE_1s'] = ADE_1s / len(loader.dataset)
            dict_metrics['ADE_2s'] = ADE_2s / len(loader.dataset)
            dict_metrics['ADE_3s'] = ADE_3s / len(loader.dataset)
            dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
            dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
            dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
            dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)

        return dict_metrics


    def load(self, directory):
        pass

    def evaluate_incremental(self):

        test_loader = DataLoader(self.data_test,
                                      batch_size=256,
                                      num_workers=8,
                                      shuffle=False
                                      )
        mem_sizes = []
        incremental_results = []
        self.mem_n2n.memory_count = torch.Tensor().cuda()
        self.mem_n2n.num_prediction = 0
        self.mem_n2n.init_memory(self.data_train)
        self.mem_n2n.num_prediction = 1
        res = self.evaluate_slim(test_loader)
        incremental_results.append(res)
        with torch.no_grad():
            for epoch in range(10):
                for step, (index, past, future, _, _, _, _, _, _, scene_one_hot) in enumerate(tqdm.tqdm(self.train_loader)):
                    mem_size = self.mem_n2n.memory_past.shape[0]
                    print('mem_size {}'.format(mem_size))
                    if mem_size <= self.config.preds:
                        self.mem_n2n.num_prediction = mem_size
                    self.iterations += 1
                    past = Variable(past)
                    future = Variable(future)
                    # scene_one_hot = Variable(scene_one_hot)
                    if self.config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                        # scene_one_hot = scene_one_hot.cuda()

                    self.mem_n2n.write_in_memory(past, future)
                    print(self.mem_n2n.memory_past.shape[0])
                    mem_sizes.append(self.mem_n2n.memory_past.shape[0])
                    if self.mem_n2n.memory_past.shape[0] == mem_size:
                        incremental_results.append(res)
                    else:
                        res = self.evaluate_slim(test_loader)
                        incremental_results.append(res)
                np.savez('incremental_results{}.npz'.format(epoch), incremental_results, mem_sizes)
                plt.plot(mem_sizes)
                plt.show()
                plt.plot([x['horizon40s'] for x in incremental_results])
                plt.show()


            # # save memory
            # torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
            # torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')

