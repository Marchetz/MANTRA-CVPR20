import os
import time
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
from torch.autograd import Variable
import csv
from models.model_argo_map import model_argo_map
import pdb
import scipy.signal
import pickle

#dataset
from dataset import dataset_argo

class Validator():
    def __init__(self, config):
        """
        class to evaluate Memnet
        :param config: configuration parameters (see test.py)
        """

        self.name_test = str(datetime.datetime.now())[:19]
        self.folder_test = 'test_results/test_argo/' + self.name_test + '_' + config.info
        if not os.path.exists(self.folder_test):
            os.makedirs(self.folder_test)
        self.folder_test = self.folder_test + '/'

        if not os.path.exists(self.folder_test + 'qualitative'):
            os.makedirs(self.folder_test + 'qualitative')

        print('creating dataset...')
        self.data_train = torch.load('data/data_train_map.pt')
        self.data_test = torch.load('data/data_val_map.pt')

        self.train_loader = DataLoader(self.data_train,
                                       batch_size=config.batch_size,
                                       num_workers=1,
                                       shuffle=True
                                       )

        self.test_loader = DataLoader(self.data_test,
                                      batch_size=1024,
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
            "future_len": config.future_len,
            "th_memory": config.th_memory
        }

        # self.model_pretrained = torch.load(config.model_pretrained)
        # self.model_controller = torch.load(config.model_controller)
        #self.mem_n2n = model_argo_map(self.settings, self.model_pretrained, self.model_controller)
        self.mem_n2n = torch.load(config.model)
        self.mem_n2n.num_prediction = config.preds
        self.EuclDistance = nn.PairwiseDistance(p=2)
        self.iterations = 0
        if config.cuda:
            self.mem_n2n = self.mem_n2n.cuda()
        self.start_epoch = 0
        self.config = config

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

    def print_tracks_memory(self):

        # print memory on tensorboard
        mem_size = self.mem_n2n.memory_past.shape[0]
        for i in range(mem_size):
            track_mem = self.mem_n2n.check_memory(i).squeeze(0).cpu().detach().numpy()
            plt.plot(track_mem[:, 0], track_mem[:, 1], marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(self.folder_test + 'memory.png')
        plt.close()


    def test_model(self):
        """
        :return: None
        """
        # populate the memory
        start = time.time()
        self._memory_writing(self.config.memory_saved)
        end = time.time()
        print('writing time: ' + str(end-start))
        print('memory_size: ' + str(len(self.mem_n2n.memory_past)))

        # run test!
        print('start test!')
        dict_metrics_test = self.evaluate(self.test_loader, 1)
        self.save_results(dict_metrics_test, epoch=0)
        #self.print_tracks_memory()

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

        self.file.write("model: with map" + '\n')
        self.file.write("num_predictions:" + str(self.config.preds) + '\n')
        self.file.write("epoch: " + str(epoch) + '\n')
        self.file.write("memory size: " + str(len(self.mem_n2n.memory_past)) + '\n')
        self.file.write("TEST size: " + str(len(self.data_test)) + '\n')

        self.file.write("error 1s: " + str(dict_metrics_test['horizon10s']) + '\n')
        self.file.write("error 3s: " + str(dict_metrics_test['horizon30s']) + '\n')
        self.file.write("ADE 1s: " + str(dict_metrics_test['ADE_1s']) + '\n')
        self.file.write("ADE 3s: " + str(dict_metrics_test['ADE_3s']) + '\n')

        self.file.close()

    def draw_track(self, past, future=None, scene_track=None, pred=None, angle=0, video_id='', vec_id='', index_tracklet=0,
                   save_fig=False, path='', horizon_dist=None, attention=None, probs=None):


        fig = plt.figure()
        plt.imshow(scene_track)
        colors = pl.cm.Reds(np.linspace(1, 0.3, pred.shape[0]))
        past = past.cpu().numpy()
        future = future.cpu().numpy()
        scene_track = scene_track.cpu().numpy()

        plt.plot(past[:, 0] * 2 + 90, past[:, 1] * 2, c='blue', linewidth=1, marker='o', markersize=1)
        if pred is not None:
            for i_p in reversed(range(pred.shape[0])):
                #pred_i = cv2.transform(pred[i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()
                pred_scene = pred[i_p].cpu().numpy()
                plt.plot(pred_scene[:, 0] * 2 + 90, pred_scene[:, 1] * 2, color=colors[i_p], linewidth=0.5, marker='o', markersize=0.5)
        if future is not None:
            plt.plot(future[:, 0] * 2 + 90, future[:, 1] * 2, c='green', linewidth=1, marker='o', markersize=1)

        if horizon_dist is not None:
            plt.title('HE 1s: ' + str(horizon_dist[0]) + ' HE 2s: ' + str(horizon_dist[1]) + ' HE 3s: ' + str(
                horizon_dist[2]) + ' HE 4s: ' + str(horizon_dist[3]))
        plt.axis('equal')
        #plt.legend()

        if save_fig:
            plt.savefig(path + str(index_tracklet).zfill(3) + '.png', format='png')
        plt.close(fig)


    def evaluate(self, loader, epoch=0):
        """
        Evaluate model. Future trajectories are predicted and
        :param loader: data loader for testing data
        :param epoch: epoch index (default: 0)
        :return: dictionary of performance metrics
        """

        index_qualitative = [1, 10000, 10014, 10017, 10019, 10025, 10031, 10036, 10080, 1009, 10093, 1012, 1013, 10130, 10155, 10165,
                             10196, 10215, 10216, 10219, 10220, 10243, 10279, 10285, 10288, 10305, 10331, 10351, 10364, 10377, 10425]

        config = self.config
        self.mem_n2n.eval()
        with torch.no_grad():
            dict_metrics = {}
            preds_dict = {}
            ADE_1s = ADE_3s = horizon10s = horizon30s = 0

            # Loop over samples
            for step, (index, past, future, presents, angle_presents, scene) in enumerate(loader):
                past = Variable(past)
                future = Variable(future)
                scene_input = Variable(torch.Tensor(np.eye(2, dtype=np.float32)[scene]).permute(0, 3, 1, 2))
                if config.cuda:
                    past = past.cuda()
                    future = future.cuda()
                    scene_input = scene_input.cuda()
                pred, attention_save = self.mem_n2n(past=past, scene=scene_input, preds_scene=self.config.preds_scene)
                pred_copy = pred.clone()
                for i in range(pred.shape[0]):
                    matRot_track = cv2.getRotationMatrix2D((0, 0), -angle_presents[i], 1)
                    for i_p in reversed(range(pred[i].shape[0])):
                        pred_copy[i][i_p] = torch.Tensor(cv2.transform(pred_copy[i][i_p].cpu().numpy().reshape(-1, 1, 2), matRot_track).squeeze()) + presents[i]
                    preds_dict[index[i].item()] = pred_copy[i].cpu().numpy()

                future_rep = future.unsqueeze(1).repeat(1, self.config.preds, 1, 1)
                distances = torch.norm(pred - future_rep, dim=3)
                mean_distances = torch.mean(distances, dim=2)
                index_min = torch.argmin(mean_distances, dim=1)
                min_distances = distances[torch.arange(0, len(index_min)), index_min]

                ADE_1s += torch.sum(torch.mean(min_distances[:, :10], 1))
                ADE_3s += torch.sum(torch.mean(min_distances[:, :30], 1))

                horizon10s += torch.sum(min_distances[:, 9])
                horizon30s += torch.sum(min_distances[:, 29])
                for i in range(len(past)):
                    if index[i].item() in index_qualitative:
                        att = attention_save[i]
                        temp = np.round(att.cpu().numpy(), 3)
                        np.savetxt(self.folder_test + 'qualitative/' + str(index[i].item()) + '_att.txt', temp, fmt='%10.5f')
                        self.draw_track(past[i], future=future[i], scene_track=scene[i], pred=pred[i], index_tracklet=index[i].item(), save_fig=True,
                                        path=self.folder_test + 'qualitative/')


            dict_metrics['ADE_1s'] = ADE_1s / len(loader.dataset)
            dict_metrics['ADE_3s'] = ADE_3s / len(loader.dataset)
            dict_metrics['horizon10s'] = horizon10s / len(loader.dataset)
            dict_metrics['horizon30s'] = horizon30s / len(loader.dataset)

        with open(self.folder_test + 'preds_evaluate.pkl', 'wb') as handle:
            pickle.dump(preds_dict, handle)
        return dict_metrics


    def _memory_writing(self, memory_saved):
        """
        writing in the memory with controller (loop over all train dataset)
        :return: loss
        """
        self.mem_n2n.memory_count = torch.Tensor().cuda()
        if memory_saved:
            self.mem_n2n.memory_past = torch.load('pretrained_models/memory_saved_argo/memory_past.pt')
            self.mem_n2n.memory_fut = torch.load('pretrained_models/memory_saved_argo/memory_fut.pt')
            self.mem_n2n.memory_scene = torch.load('pretrained_models/memory_saved_argo/memory_scene.pt')
        else:
            self.mem_n2n.init_memory(self.data_train)
            config = self.config
            with torch.no_grad():
                for step, (index, past, future, presents, angle_presents, scene) in enumerate(self.train_loader):
                    past = Variable(past)
                    future = Variable(future)
                    scene_input = Variable(torch.Tensor(np.eye(2, dtype=np.float32)[scene]).permute(0, 3, 1, 2))
                    if config.cuda:
                        past = past.cuda()
                        future = future.cuda()
                        scene_input = scene_input.cuda()
                    self.mem_n2n.write_in_memory(past, future, scene_input)
        #save memory
        torch.save(self.mem_n2n.memory_past, self.folder_test + 'memory_past.pt')
        torch.save(self.mem_n2n.memory_fut, self.folder_test + 'memory_fut.pt')
        torch.save(self.mem_n2n.memory_scene, self.folder_test + 'memory_scene.pt')

    def load(self, directory):
        pass
