import os
import json
import dataset
from torch.utils.data import DataLoader
import datetime
import time

name_test = str(datetime.datetime.now())[:19]
folder_test = 'test/' + name_test
if not os.path.exists(folder_test):
    os.makedirs(folder_test)
folder_test = folder_test + '/'

tracks = json.load(open("world_traj_kitti.json"))
dim_clip = 180
past_len = 20
future_len = 40

start = time.time()
print('Creating dataset...')
data_train = dataset.TrackDataset(tracks,
                                  num_instances=past_len,
                                  num_labels=future_len,
                                  train=True,
                                  dim_clip=dim_clip)

data_test = dataset.TrackDataset(tracks,
                                 num_instances=past_len,
                                 num_labels=future_len,
                                 train=False,
                                 dim_clip=dim_clip)
print('Dataset created')
end = time.time()
print('time: ' + str(end - start))

train_loader = DataLoader(data_train,
                          batch_size=32,
                          num_workers=1,
                          shuffle=True
                          )
test_loader = DataLoader(data_test,
                         batch_size=32,
                         num_workers=1,
                         shuffle=False
                         )

# save dataset in a folder
data_train.save_dataset(folder_test)
data_test.save_dataset(folder_test)

# save scene of all videos with all trajectories
data_train.save_scenes_with_tracks(folder_test)
data_test.save_scenes_with_tracks(folder_test)
