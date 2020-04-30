import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
import cv2
import math
import pdb

# colormap
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
    def __init__(self, tracks, len_past=20, len_future=40, train=False, dim_clip=180):

        self.tracks = tracks      # dataset dict
        self.dim_clip = dim_clip  # dim_clip*2 is the dimension of scene (pixel)
        self.is_train = train

        self.video_track = []     # '0001'
        self.vehicles = []        # 'Car'
        self.number_vec = []      # '4'
        self.index = []           # '50'
        self.pasts = []           # [len_past, 2]
        self.presents = []        # position in complete scene
        self.angle_presents = []  # trajectory angle in complete scene
        self.futures = []         # [len_future, 2]
        self.scene = []           # [dim_clip, dim_clip, 1], scene fot qualitative examples
        self.scene_crop = []      # [dim_clip, dim_clip, 4], input to IRM

        num_total = len_past + len_future
        self.video_split, self.ids_split_test = self.get_desire_track_files(train)

        # Preload data
        for video in self.video_split:
            vehicles = self.tracks[video].keys()
            video_id = video[-9:-5]
            print('video: ' + video_id)
            path_scene = 'maps/2011_09_26__2011_09_26_drive_' + video_id + '_sync_map.png'
            scene_track = cv2.imread(path_scene, 0)
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
                for count in range(0, len_track, 1):
                    if len_track - count > num_total:

                        temp_past = points[count:count + len_past].copy()
                        temp_future = points[count + len_past:count + num_total].copy()
                        origin = temp_past[-1]

                        # filter out noise for non-moving vehicles
                        if np.var(temp_past[:, 0]) < 0.1 and np.var(temp_past[:, 1]) < 0.1:
                            temp_past = np.zeros((20, 2))
                        else:
                            temp_past = temp_past - origin

                        if np.var(temp_past[:, 0]) < 0.1 and np.var(temp_past[:, 1]) < 0.1:
                            temp_future = np.zeros((40, 2))
                        else:
                            temp_future = temp_future - origin

                        scene_track_clip = scene_track[
                                           int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                           int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                        scene_track_onehot_clip = scene_track_onehot[
                                                  int(origin[1]) * 2 - self.dim_clip:int(origin[1]) * 2 + self.dim_clip,
                                                  int(origin[0]) * 2 - self.dim_clip:int(origin[0]) * 2 + self.dim_clip]

                        # rotation invariance
                        unit_y_axis = torch.Tensor([0, -1])
                        vector = temp_past[-5]
                        if vector[0] > 0.0:
                            angle = np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                        else:
                            angle = -np.rad2deg(self.angle_vectors(vector, unit_y_axis))
                        matRot_track = cv2.getRotationMatrix2D((0, 0), angle, 1)
                        matRot_scene = cv2.getRotationMatrix2D((self.dim_clip, self.dim_clip), angle, 1)

                        past_rot = cv2.transform(temp_past.reshape(-1, 1, 2), matRot_track).squeeze()
                        future_rot = cv2.transform(temp_future.reshape(-1, 1, 2), matRot_track).squeeze()
                        scene_track_onehot_clip = cv2.warpAffine(scene_track_onehot_clip, matRot_scene,
                                           (scene_track_onehot_clip.shape[0], scene_track_onehot_clip.shape[1]),
                                           borderValue=0,
                                           flags=cv2.INTER_NEAREST)  # (1, 0, 0, 0)

                        self.index.append(count + 19 + start_frame)
                        self.pasts.append(past_rot)
                        self.futures.append(future_rot)
                        self.presents.append(origin)
                        self.angle_presents.append(angle)
                        self.video_track.append(video_id)
                        self.vehicles.append(class_vec)
                        self.number_vec.append(num_vec)
                        self.scene.append(scene_track_clip)
                        self.scene_crop.append(scene_track_onehot_clip)

        self.index = np.array(self.index)
        self.pasts = torch.FloatTensor(self.pasts)
        self.futures = torch.FloatTensor(self.futures)
        self.presents = torch.FloatTensor(self.presents)
        self.video_track = np.array(self.video_track)
        self.vehicles = np.array(self.vehicles)
        self.number_vec = np.array(self.number_vec)
        self.scene = np.array(self.scene)

    def save_dataset(self, folder_save):
        for i in range(self.pasts.shape[0]):
            video = self.video_track[i]
            vehicle = self.vehicles[i]
            number = self.number_vec[i]
            past = self.pasts[i]
            future = self.futures[i]
            scene_track = self.scene_crop[i]

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
                    self.draw_track_in_scene(past, scene_track, index_tracklet=self.index[i], future=future, path=vehicle_path)

    def draw_track(self, past, future, index_tracklet, path):
        plt.plot(past[:, 0], -past[:, 1], c='blue', marker='o', markersize=1)
        if future is not None:
            future = future.cpu().numpy()
            plt.plot(future[:, 0], -future[:, 1], c='green', marker='o', markersize=1)
        plt.axis('equal')
        plt.savefig(path + str(index_tracklet) + '.png')
        plt.close()

    def draw_scene(self, scene_track, index_tracklet, path):
        # print semantic map
        cv2.imwrite(path + str(index_tracklet) + '.png', scene_track)

    def draw_track_in_scene(self, story, scene_track, index_tracklet, future=None, path=''):
        plt.imshow(scene_track, cmap=cm)
        plt.plot(story[:, 0] * 2 + self.dim_clip, story[:, 1] * 2 + self.dim_clip, c='blue', marker='o', markersize=1)
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
        if train:
            desire_ids = [5, 9, 11, 13, 14, 17, 27, 28, 48, 51, 56, 57, 59, 60, 84, 91]
        else:
            desire_ids = [1, 2, 15, 18, 29, 32, 52, 70]

        tracklet_files = ['video_2011_09_26__2011_09_26_drive_' + str(x).zfill(4) + '_sync'
                          for x in desire_ids]
        return tracklet_files, desire_ids

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_vectors(self, v1, v2):
        """ Returns angle between two vectors.  """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
        if math.isnan(angle):
            return 0.0
        else:
            return angle

    def __len__(self):
        return self.pasts.shape[0]

    def __getitem__(self, idx):
        return self.index[idx], self.pasts[idx], self.futures[idx], self.presents[idx], self.angle_presents[idx], self.video_track[idx], \
               self.vehicles[idx], self.number_vec[idx], self.scene[idx], np.eye(4, dtype=np.float32)[self.scene_crop[idx]]


