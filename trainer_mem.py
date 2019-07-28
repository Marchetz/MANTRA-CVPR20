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
from models.model_memory_single import model_memory_single
import dataset
import test_index


class Trainer():
    def __init__(self, config):
        """
        Trainer class for training the iterative refinement module (MRI)
        :param config: configuration parameters (see train_MRI.py)
        """

        self.test_index = test_index.dict_test
        self.name_run = 'runs-mem-RNNSCENE/'
        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'test/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'
        self.file = open(self.folder_test + "details.txt", "w")
        tracks = json.load(open("world_traj_kitti_with_intervals_correct.json"))

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
        self.model_pretrained = torch.load(config.model)
        self.mem_n2n = model_memory_single(self.settings, self.model_pretrained)
        self.mem_n2n.past_len = config.past_len
        self.mem_n2n.future_len = config.future_len

        self.criterionLoss = nn.MSELoss()
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
        self.writer = SummaryWriter(self.name_run + self.name_test + '_' + config.info)
        self.writer.add_text('Training Configuration', 'model name: ' + self.mem_n2n.name_model, 0)
        self.writer.add_text('Training Configuration', 'dataset train: ' + str(len(self.data_train)), 0)
        self.writer.add_text('Training Configuration', 'dataset test: ' + str(len(self.data_test)), 0)
        self.writer.add_text('Training Configuration', 'number of prediction: ' + str(self.num_prediction), 0)
        self.writer.add_text('Training Configuration', 'rotation_aug: ' + str(config.rotation_aug), 0)
        self.writer.add_text('Training Configuration', 'batch_size: ' + str(self.config.batch_size), 0)
        self.writer.add_text('Training Configuration', 'learning rate init: ' + str(self.config.learning_rate), 0)
        self.writer.add_text('Training Configuration', 'dim_embedding_key: ' + str(self.settings["dim_embedding_key"]), 0)

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
        for param in self.mem_n2n.decoder.parameters():
            param.requires_grad = False
        for param in self.mem_n2n.FC_output.parameters():
            param.requires_grad = False

        # Load memory
        print('loading memory')
        self.mem_n2n.memory_past = torch.load(config.memory_past)
        self.mem_n2n.memory_fut = torch.load(config.memory_fut)
        print('memory updated')

        self.mem_n2n.memory_count = np.zeros((len(self.mem_n2n.memory_past), 1))

        step_results = [1, 10, 20, 40, 50, 60, 80, 100, 150, 200, 250, 300, 350, 400, 450, 490, 550, 600]

        # Main training loop
        for epoch in range(self.start_epoch, config.max_epochs):

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
                self.writer.add_scalar('accuracy_test/Horizon01s', dict_metrics_test['horizon01s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon10s', dict_metrics_test['horizon10s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon20s', dict_metrics_test['horizon20s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon30s', dict_metrics_test['horizon30s'], epoch)
                self.writer.add_scalar('accuracy_test/Horizon40s', dict_metrics_test['horizon40s'], epoch)
                self.writer.add_scalar('dimension_memory/memory', len(self.mem_n2n.memory_past), epoch)

                # Save model checkpoint
                torch.save(self.mem_n2n, self.folder_test + 'model' + self.name_test)

                self.save_results(dict_metrics_test, epoch=epoch + 1)

            for name, param in self.mem_n2n.named_parameters():
                self.writer.add_histogram(name, param.data, epoch)

    def save_results(self, dict_metrics_test, dict_metrics_train=None, epoch=0):
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

    def draw_track(self, past, future, scene_track, pred=None, video_id='', vec_id='', index_tracklet=0,
                   save_fig=False, path='', remove_pred=''):

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
                plt.plot(pred_scene[:, 0], pred_scene[:, 1], color=colors[i_p], linewidth=0.5, marker='o', markersize=0.5)

        plt.plot(future_scene[:, 0], future_scene[:, 1], c='green', linewidth=1, marker='o', markersize=1)

        plt.title('video: ' + video_id + ', vehicle: ' + vec_id + ',index: ' + str(index_tracklet))
        plt.axis('equal')

        if save_fig:
            plt.savefig(path + video_id + '_' + vec_id + '_' + str(index_tracklet).zfill(3) + remove_pred + '.png')
        plt.close(fig)

    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
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

            for step, (index, past, future, presents, videos, vehicles, number_vec, scene, scene_one_hot) in enumerate(loader):
                past = Variable(past)
                future = Variable(future)
                scene_one_hot = Variable(scene_one_hot)
                if self.config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    scene_one_hot = scene_one_hot.cuda()
                pred = self.mem_n2n(past, scene_one_hot)

                # salva img con tutte le predizioni e img con quelle rimosse
                for i in range(len(past)):
                    scene_i = scene[i]
                    pred_good = torch.Tensor().cuda()
                    listError = []

                    for i_multiple in range(len(pred[i])):
                        pred_one = pred[i][i_multiple]
                        dist = self.EuclDistance(pred_one, future[i, :])
                        listError.append(torch.mean(dist))
                        # error_scene = 0
                        pred_one = pred[i][i_multiple]
                        pred_one_scene = pred_one * 2 + self.dim_clip
                        pred_one_scene = pred_one_scene.type(torch.LongTensor)

                        if 1 in (pred_one_scene > 359).reshape(-1):
                            pred_situation = torch.ones(20)
                        else:
                            pred_situation = scene_i[pred_one_scene[:, 1], pred_one_scene[:, 0]]

                        # Count points predicted outside from the road
                        error_scene = len(np.where(pred_situation != 1)[0])
                        if error_scene < 10:
                            pred_good = torch.cat((pred_good, pred_one.unsqueeze(0)), 0)
                        # #scarta la predizione se è sopra di un certo errore
                        # #faccio un nuova variabile e ci appendo quelle giuste

                    if len(pred_good) == 0:
                        pred_one = pred[i][0]
                        pred_good = torch.cat((pred_good, pred_one.unsqueeze(0)), 0)

                    i_min = np.argmin(listError)
                    dist = self.EuclDistance(pred[i][i_min], future[i, :])

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
                        with open(self.folder_test + 'preds_test.json', 'w') as outfile:
                            json.dump(videos_json, outfile)

                        if self.config.saveImages_All:
                            if not os.path.exists(self.folder_test + vid):
                                os.makedirs(self.folder_test + vid)
                            video_path = self.folder_test + vid + '/'
                            if not os.path.exists(video_path + vec + num_vec):
                                os.makedirs(video_path + vec + num_vec)
                            vehicle_path = video_path + vec + num_vec + '/'
                            # prima della cancellazione
                            # self.draw_track(past[i], future[i], scene[i], pred[i], present, vid, vec+num_vec, index_tracklet=index_track, num_epoch=epoch, saveFig=True,
                            #                path=vehicle_path, remove_pred = '')
                            # dopo la cancellazione
                            self.draw_track(past[i], future[i], scene[i], pred_good, present, vid, vec + num_vec,
                                            index_tracklet=index_track, num_epoch=epoch, save_fig=True,
                                            path=vehicle_path, remove_pred='_Remove')
                            # #prime 10 traiettorie nel ranking
                            # self.draw_track(past[i], future[i], scene[i], pred_good, present, vid, index_tracklet=index[i], num_epoch=epoch, saveFig=True,
                            #                 path=vehicle_path, remove_pred = '_2Better', index_ranking=i_min)
                        else:
                            # #pdb.set_trace()
                            if index_track.item() in self.test_index[vid][vec + num_vec]:
                                # fai una cartella dove metti gli esempi salienti
                                if not os.path.exists(self.folder_test + 'highlights'):
                                    os.makedirs(self.folder_test + 'highlights')
                                highlights_path = self.folder_test + 'highlights' + '/'
                                # prima della cancellazione
                                self.draw_track(past[i], future[i], scene[i], pred[i], present, vid, vec + num_vec,
                                                index_tracklet=index_track, num_epoch=epoch, save_fig=True,
                                                path=highlights_path, remove_pred='_0noRemove')
                                # dopo la cancellazione
                                self.draw_track(past[i], future[i], scene[i], pred_good, present, vid, vec + num_vec,
                                                index_tracklet=index_track, num_epoch=epoch, save_fig=True,
                                                path=highlights_path, remove_pred='_1YesRemove')

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

    def _train_single_epoch(self):
        config = self.config
        for step, (index, past, future, presents, videos, vehicles, number_vec, scene, scene_one_hot) in enumerate(
                self.train_loader):
            self.iterations += 1
            if config.rotation_aug:
                for i_rotate in range(len(past)):
                    angle = randint(0, 360)
                    rot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
                    rot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)
                    past[i_rotate] = torch.FloatTensor(
                        cv2.transform(past[i_rotate].numpy().reshape(-1, 1, 2), rot_track).squeeze())
                    future[i_rotate] = torch.FloatTensor(
                        cv2.transform(future[i_rotate].numpy().reshape(-1, 1, 2), rot_track).squeeze())
                    # scene[i_rotate] = torch.LongTensor(cv2.warpAffine(scene[i_rotate].numpy(), rot_scene, (scene[i_rotate].shape[0], scene[i_rotate].shape[1]), borderValue=0, flags=cv2.INTER_NEAREST))
                    scene_one_hot[i_rotate] = torch.LongTensor(
                        cv2.warpAffine(scene_one_hot[i_rotate].numpy(), rot_scene,
                                       (scene[i_rotate].shape[0], scene[i_rotate].shape[1]), borderValue=(1, 0, 0, 0),
                                       flags=cv2.INTER_NEAREST))

            past = Variable(past)
            future = Variable(future)
            scene_one_hot = Variable(scene_one_hot)

            # pdb.set_trace()
            if config.cuda:
                past = past.cuda()
                future = future.cuda()
                scene_one_hot = scene_one_hot.cuda()
            self.opt.zero_grad()

            output = self.mem_n2n(past, scene_one_hot)

            prediction_better = torch.Tensor().cuda()
            for i in range(len(past)):
                list_error = []
                for i_multiple in range(len(output[i])):
                    pred_one = output[i][i_multiple]
                    dist = self.EuclDistance(pred_one, future[i, :])
                    list_error.append(torch.mean(dist))
                i_min = np.argmin(list_error)
                prediction_better = torch.cat((prediction_better, output[i][i_min].unsqueeze(0)), 0)

            # compute loss with best predicted trajectory
            loss = self.criterionLoss(prediction_better, future)

            loss.backward()
            self.opt.step()
            self.writer.add_scalar('loss/loss_total', loss, self.iterations)

        return loss.item()

    def save_dataset(self):
        self.data_test.save_dataset(self.folder_test)

    def scene_complete_tracks(self):
        self.data_train.save_scenes_with_tracks(self.folder_test)
        self.data_test.save_scenes_with_tracks(self.folder_test)
