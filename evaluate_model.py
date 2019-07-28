import os
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import LinearSegmentedColormap
import cv2
import numpy as np
import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import dataset
import test_index
from tqdm import tqdm


class Validator():
    def __init__(self, config):

        self.mem_n2n = torch.load(config.model)
        self.test_index = test_index.dict_test
        self.name_test = config.model
        self.folder_test = 'test/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        tracks = json.load(open(config.track_file))

        self.dim_clip = 180
        print('creating dataset...')

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

        self.num_prediction = config.preds
        self.settings = {
            "batch_size": config.batch_size,
            "use_cuda": config.cuda,
            "dim_embedding_key": config.dim_embedding_key,
            "past_len": config.past_len,
            "future_len": config.future_len,
            "num_prediction": self.num_prediction
        }

        # load pretrained model and create memory_model
        self.mem_n2n = torch.load(config.model)
        self.mem_n2n.num_prediction = config.preds
        self.mem_n2n.future_len = config.future_len
        self.mem_n2n.past_len = config.past_len

        self.eucl_distance = nn.PairwiseDistance(p=2)
        self.iterations = 0
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

    def test_model(self):
        """
        Perform test step and save results
        :return: None
        """
        self.mem_n2n.memory_past = torch.load('memory_past_test.pt')
        self.mem_n2n.memory_fut = torch.load('memory_fut_test.pt')
        dict_metrics_test = self.evaluate(self.test_loader, 1)
        self.save_results(dict_metrics_test, epoch=0)

    def save_results(self, dict_metrics_test, dict_metrics_train=None, epoch=0):
        """
        Save test results
        :param dict_metrics_test: dictionary with test metrics
        :param dict_metrics_train: dictionary with test metrics
        :param epoch: epoch number (default: 0)
        :return: None
        """
        self.file = open(self.folder_test + "results.txt", "w")
        self.file.write("TEST:" + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("split test: " + str(self.data_test.ids_split_test) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')

        self.file.write("error 1s: " + str(dict_metrics_test['horizon10s'].item()) + '\n')
        self.file.write("error 2s: " + str(dict_metrics_test['horizon20s'].item()) + '\n')
        self.file.write("error 3s: " + str(dict_metrics_test['horizon30s'].item()) + '\n')
        self.file.write("error 4s: " + str(dict_metrics_test['horizon40s'].item()) + '\n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s'].item()) + '\n')
        self.file.write("ADE 2s: " + str(dict_metrics_test['ADE_2s'].item()) + '\n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s'].item()) + '\n')
        self.file.write("ADE 4s: " + str(dict_metrics_test['euclMean'].item()) + '\n')

        if dict_metrics_train is not None:
            self.file.write("TRAIN:" + '\n')
            self.file.write("error 1s: " + str(dict_metrics_train['horizon10s'].item()) + '\n')
            self.file.write("error 2s: " + str(dict_metrics_train['horizon20s'].item()) + '\n')
            self.file.write("error 3s: " + str(dict_metrics_train['horizon30s'].item()) + '\n')
            self.file.write("error 4s: " + str(dict_metrics_train['horizon40s'].item()) + '\n')
            self.file.write("ADE 1s: " + str(dict_metrics_train['ADE_1s'].item()) + '\n')
            self.file.write("ADE 2s: " + str(dict_metrics_train['ADE_2s'].item()) + '\n')
            self.file.write("ADE 3s: " + str(dict_metrics_train['ADE_3s'].item()) + '\n')
            self.file.write("ADE 4s: " + str(dict_metrics_train['euclMean'].item()) + '\n')
        self.file.close()

    def draw_track(self, past, future, scene_track, pred=None, present=None, video_id='', vec_id='', index_tracklet=0,
                   num_epoch=0, saveFig=False, path='', remove_pred='', index_ranking=None):

        colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.49, 0.33, 0.16), (0.29, 0.57, 0.25)]
        cmap_name = 'scene_cmap'
        cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=5)
        fig = plt.figure()

        plt.imshow(scene_track, cmap=cm)
        colors = pl.cm.Reds(np.linspace(1, 0.3, pred.shape[0]))

        past = past.cpu().numpy()
        future = future.cpu().numpy()

        story_scene = past * 2 + self.dim_clip
        future_scene = future * 2 + self.dim_clip

        plt.plot(story_scene[:, 0], story_scene[:, 1], c='blue', linewidth=1, marker='o', markersize=1)

        if pred is not None:
            for i_p in reversed(range(pred.shape[0])):
                pred_i = pred[i_p].cpu().numpy()
                pred_scene = pred_i * 2 + self.dim_clip
                plt.plot(pred_scene[:, 0], pred_scene[:, 1], color=colors[i_p], linewidth=0.5, marker='o',
                         markersize=0.5)

        plt.plot(future_scene[:, 0], future_scene[:, 1], c='green', linewidth=1, marker='o', markersize=1)

        plt.title('video: ' + video_id + ', vehicle: ' + vec_id + ',index: ' + str(index_tracklet))
        plt.axis('equal')

        if saveFig:
            plt.savefig(path + video_id + '_' + vec_id + '_' + str(index_tracklet).zfill(3) + remove_pred + '.png')

        plt.close(fig)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate model
        :param loader: data loader for test samples
        :param epoch: epoch number (default: 0)
        :return: metric dictionary
        """

        videos_json = {}
        test_ids = self.data_test.ids_split_test

        for ids in test_ids:
            videos_json['video_' + str(ids).zfill(4)] = {}
            for i_temp in range(self.data_test.video_length[str(ids).zfill(4)]):
                videos_json['video_' + str(ids).zfill(4)]['frame_' + str(i_temp).zfill(3)] = {}

        with torch.no_grad():
            dict_metrics = {}
            eucl_mean = ADE_1s = ADE_2s = ADE_3s = horizon01s = horizon10s = horizon20s = horizon30s = horizon40s = 0

            list_err1 = []
            list_err2 = []
            list_err3 = []
            list_err4 = []

            for step, (index, past, future, presents, videos, vehicles, number_vec, scene, scene_one_hot) in tqdm(enumerate(loader)):
                past = Variable(past)
                future = Variable(future)
                scene_one_hot = Variable(scene_one_hot)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    scene_one_hot = scene_one_hot.cuda()
                pred = self.mem_n2n(past, scene_one_hot)

                for i in range(len(past)):
                    scene_i = scene[i]
                    pred_good = torch.Tensor().cuda()
                    list_error = []

                    for i_multiple in range(len(pred[i])):

                        pred_one = pred[i][i_multiple]
                        dist = self.eucl_distance(pred_one, future[i, :])
                        list_error.append(torch.mean(dist))
                        pred_one = pred[i][i_multiple]
                        pred_one_scene = pred_one * 2 + self.dim_clip
                        pred_one_scene = pred_one_scene.type(torch.LongTensor)

                        if 1 in (pred_one_scene > 359).reshape(-1):
                            pred_situation = torch.ones(20)
                        else:
                            pred_situation = scene_i[pred_one_scene[:, 1], pred_one_scene[:, 0]]

                        # Count points predicted outside from the road and remove bad predictions
                        error_scene = len(np.where(pred_situation != 1)[0])
                        if error_scene < 10:
                            pred_good = torch.cat((pred_good, pred_one.unsqueeze(0)), 0)

                    if len(pred_good) == 0:
                        pred_one = pred[i][0]
                        pred_good = torch.cat((pred_good, pred_one.unsqueeze(0)), 0)

                    i_min = np.argmin(list_error)
                    dist = self.eucl_distance(pred[i][i_min], future[i, :])

                    eucl_mean += torch.mean(dist)
                    ADE_1s += torch.mean(dist[:10])
                    ADE_2s += torch.mean(dist[:20])
                    ADE_3s += torch.mean(dist[:30])
                    list_err1.append(dist[9].cpu())
                    list_err2.append(dist[19].cpu())
                    list_err3.append(dist[29].cpu())
                    list_err4.append(dist[39].cpu())
                    horizon01s += dist[0]
                    horizon10s += dist[9]
                    horizon20s += dist[19]
                    horizon30s += dist[29]
                    horizon40s += dist[39]

                    vid = videos[i]
                    vec = vehicles[i]
                    num_vec = number_vec[i]
                    index_track = index[i].numpy()
                    present = presents[i].cpu()

                    # json
                    st = past[i].cpu() + present
                    fu = future[i].cpu() + present
                    pr = pred_good.cpu() + present
                    videos_json['video_' + vid]['frame_' + str(index_track).zfill(3)][vec + num_vec] = {}
                    videos_json['video_' + vid]['frame_' + str(index_track).zfill(3)][vec + num_vec][
                        'pred'] = pr.tolist()
                    videos_json['video_' + vid]['frame_' + str(index_track).zfill(3)][vec + num_vec][
                        'past'] = st.tolist()
                    videos_json['video_' + vid]['frame_' + str(index_track).zfill(3)][vec + num_vec][
                        'fut'] = fu.tolist()
                    videos_json['video_' + vid]['frame_' + str(index_track).zfill(3)][vec + num_vec][
                        'present'] = present.tolist()

                    if loader == self.test_loader and self.config.saveImages:
                        if self.config.saveImages_All:
                            if not os.path.exists(self.folder_test + vid):
                                os.makedirs(self.folder_test + vid)
                            video_path = self.folder_test + vid + '/'
                            if not os.path.exists(video_path + vec + num_vec):
                                os.makedirs(video_path + vec + num_vec)
                            vehicle_path = video_path + vec + num_vec + '/'

                            self.draw_track(past[i], future[i], scene[i], pred_good, present, vid, vec + num_vec,
                                            index_tracklet=index_track, num_epoch=epoch, saveFig=True,
                                            path=vehicle_path, remove_pred='_Remove')
                        else:
                            if index_track.item() in self.test_index[vid][vec + num_vec]:
                                # Save interesting results
                                if not os.path.exists(self.folder_test + 'highlights'):
                                    os.makedirs(self.folder_test + 'highlights')
                                highlights_path = self.folder_test + 'highlights' + '/'
                                # before removing bad predictions
                                self.draw_track(past[i], future[i], scene[i], pred[i], present, vid, vec + num_vec,
                                                index_tracklet=index_track, num_epoch=epoch, saveFig=True,
                                                path=highlights_path, remove_pred='_0noRemove')
                                # after removing bad predictions
                                self.draw_track(past[i], future[i], scene[i], pred_good, present, vid, vec + num_vec,
                                                index_tracklet=index_track, num_epoch=epoch, saveFig=True,
                                                path=highlights_path, remove_pred='_1YesRemove')

            with open(self.folder_test + 'preds_test.json', 'w') as outfile:
                json.dump(videos_json, outfile)

            dict_metrics['eucl_mean'] = eucl_mean / len(loader.dataset)
            dict_metrics['ADE_1s'] = ADE_1s / len(loader.dataset)
            dict_metrics['ADE_2s'] = ADE_2s / len(loader.dataset)
            dict_metrics['ADE_3s'] = ADE_3s / len(loader.dataset)
            dict_metrics['horizon01s'] = horizon01s / len(loader.dataset)
            dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
            dict_metrics['horizon20s'] = horizon20s / len(loader.dataset)
            dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)
            dict_metrics['horizon40s'] = horizon40s / len(loader.dataset)

        return dict_metrics
