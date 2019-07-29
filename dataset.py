import numpy as np
import torch
import torch.utils.data as data
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import cv2

colors = [(0, 0, 0), (0.87, 0.87, 0.87), (0.54, 0.54, 0.54), (0.29, 0.57, 0.25)]
cmap_name = 'scene_list'
cm = LinearSegmentedColormap.from_list(
    cmap_name, colors, N=4)


class TrackDataset(data.Dataset):
    """
    Dataset class for KITTI.
    The building class is merged into the background class
    0:background 1:street 2:sidewalk, 3:building 4: vegetation ---> 0:background 1:street 2:sidewalk, 3: vegetation
    """
    def __init__(self, tracks, num_instances, num_labels, train, dim_clip):

        self.tracks = tracks
        self.dim_clip = dim_clip
        self.video_length = {}
        self.is_train = train

        self.video_track = []   # '0001'
        self.vehicles = []      # 'Car'
        self.number_vec = []    # '4'
        self.index = []         # '50'
        self.instances = []     # [num_instances, 2]
        self.presents = []      # position in complete scene
        self.labels = []        # [num_labels, 2]
        self.scene = []         # [dim_clip, dim_clip, 1]
        self.scene_crop = []    # [dim_clip, dim_clip, 4]

        num_total = num_instances + num_labels
        self.video_split, self.ids_split_test = self.get_desire_track_files(train)

        # Preload data
        for video in self.video_split:
            vehicles = self.tracks[video].keys()
            video_id = video[-9:-5]
            track_ego = np.array(self.tracks[video]['track_0']['trajectory']).T
            print('video: ' + video_id)
            path_scene = 'maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png'
            scene_track = cv2.imread(path_scene, 0) - 1

            scene_track_onehot = scene_track.copy()
            # Remove building class
            scene_track_onehot[np.where(scene_track_onehot == 3)] = 0
            scene_track_onehot[np.where(scene_track_onehot == 4)] -= 1

            for vec in vehicles:
                class_vec = tracks[video][vec]['cls']
                num_vec = vec.split('_')[1]
                start_frame = tracks[video][vec]['start']
                points = np.array(tracks[video][vec]['trajectory']).T
                len_track = len(points)
                self.video_length[video_id] = len(track_ego)
                for count in range(0, len_track, 1):

                    if len_track - count > num_total:
                        temp_istance = points[count:count + num_instances].copy()
                        temp_label = points[count + num_instances:count + num_total].copy()

                        origin = temp_istance[-1]
                        if np.var(temp_istance[:, 0]) < 0.1 and np.var(temp_istance[:, 1]) < 0.1:
                            st = np.zeros((20, 2))
                        else:
                            st = temp_istance - origin

                        if np.var(temp_istance[:, 0]) < 0.1 and np.var(temp_istance[:, 1]) < 0.1:
                            fu = np.zeros((40, 2))
                        else:
                            fu = temp_label - origin

                        scene_track_clip = scene_track[
                                           int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                           int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                        scene_track_onehot_clip = scene_track_onehot[
                                                  int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                                  int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                        self.index.append(count + 19 + start_frame)
                        self.instances.append(st)
                        self.labels.append(fu)
                        self.presents.append(origin)
                        self.video_track.append(video_id)
                        self.vehicles.append(class_vec)
                        self.number_vec.append(num_vec)
                        self.scene.append(scene_track_clip)
                        self.scene_crop.append(scene_track_onehot_clip)

        self.index = np.array(self.index)
        self.instances = torch.FloatTensor(self.instances)
        self.labels = torch.FloatTensor(self.labels)
        self.presents = torch.FloatTensor(self.presents)

        self.video_track = np.array(self.video_track)
        self.vehicles = np.array(self.vehicles)
        self.number_vec = np.array(self.number_vec)
        self.scene = np.array(self.scene)

    def save_scenes_with_tracks(self, folder_save):
        for video in self.video_split:
            fig = plt.figure()
            video_id = video[-9:-5]
            im = plt.imread('maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png')
            implot = plt.imshow(im, cmap=cm)
            for t in self.tracks[video].keys():
                points = np.array(self.tracks[video][t]['trajectory']).T
                if len(points.shape) > 1:
                    plt.plot(points[:, 0] * 2, points[:, 1] * 2)
            plt.savefig(folder_save + video_id + '.png')
            plt.close(fig)

    def save_dataset(self, folder_save):
        for i in range(len(self.instances)):
            video = self.video_track[i]
            vehicle = self.vehicles[i]
            number = self.number_vec[i]
            past = self.instances[i]
            future = self.labels[i]
            scene_track = self.scene[i]

            saving_list = ['only_tracks', 'only_scenes', 'tracks_on_scene']

            for sav in saving_list:
                folder_save_type = folder_save + sav + '/'
                if not os.path.exists(folder_save_type + video):
                    os.makedirs(folder_save_type + video)
                video_path = folder_save_type + video + '/'
                if not os.path.exists(video_path + vehicle + number):
                    os.makedirs(video_path + vehicle + number)
                vehicle_path = video_path + '/' + vehicle + number + '/'
                if sav == 'only_tracks':
                    self.draw_track(past, future, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'only_scenes':
                    self.draw_scene(scene_track, index_tracklet=self.index[i], path=vehicle_path)
                if sav == 'tracks_on_scene':
                    self.draw_scene_with_track(past, scene_track, index_tracklet=self.index[i], future=future, path=vehicle_path)

    def draw_track(self, past, future, index_tracklet, path):
        past = past.cpu().numpy()
        plt.plot(past[:, 0], -past[:, 1], c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1], c='green', marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def draw_scene(self, scene_track, index_tracklet, path):
        cv2.imwrite(path + str(index_tracklet) + '.png', scene_track)

    def draw_scene_with_track(self, story, scene_track, index_tracklet, future=None, path=''):
        plt.imshow(scene_track, cmap=cm)
        story = story.cpu().numpy()
        plt.plot(story[:, 0] * 2 + self.dim_clip, story[:, 1] * 2 + self.dim_clip, c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0] * 2 + self.dim_clip, future[:, 1] * 2 + self.dim_clip, c='green', marker='o', markersize=1)
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    @staticmethod
    def get_desire_track_files(train):
        """ Get videos only from the splits defined in DESIRE: https://arxiv.org/abs/1704.04394
        Splits obtained from the authors:
        all: [1, 2, 5, 9, 11, 13, 14, 15, 17, 18, 27, 28, 29, 32, 48, 51, 52, 56, 57, 59, 60, 70, 84, 91]
        train: [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        test: [1, 2, 15, 18, 29, 32, 52, 70]
        """
        # change: 0005 <-> 0029

        if train:
            desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
            # desire_ids = [ 9, 11, 13, 14, 17, 27, 28, 29, 48, 51, 56, 57, 59, 60, 84, 91]
        else:
            desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]
            # desire_ids = [1, 2, 5, 15, 18, 32, 52, 70]

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in desire_ids]
        return tracklet_files, desire_ids

    def __getitem__(self, idx):
        return self.index[idx], self.instances[idx], self.labels[idx], self.presents[idx], self.video_track[idx], \
               self.vehicles[idx], self.number_vec[idx], self.scene[idx], to_categorical(self.scene_crop[idx], 4)

    def __len__(self):
        return len(self.instances)
