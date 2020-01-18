import torch
import torch.nn as nn
import pdb

class model_ae_map_v2(nn.Module):
    """
    Encoder-Decoder model. The model reconstructs the future trajectory from an encoding of both past and future.
    Past and future trajectories are encoded separately.
    A trajectory is first convolved with a 1D kernel and are then encoded with a Gated Recurrent Unit (GRU).
    Encoded states are concatenated and decoded with a GRU and a fully connected layer.
    The decoding process decodes the trajectory step by step, predicting offsets to be added to the previous point.
    """
    def __init__(self, settings):
        super(model_ae_map_v2, self).__init__()

        self.name_model = 'pre_train'
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        channel_in = 2
        channel_out = 16
        dim_kernel = 3
        input_gru = channel_out

        # temporal encoding
        self.conv_past = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.conv_fut = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)

        # encoder-decoder
        self.encoder_past = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
        self.encoder_fut = nn.GRU(input_gru, self.dim_embedding_key, 1, batch_first=True)
        self.decoder = nn.GRU(self.dim_embedding_key * 3, self.dim_embedding_key * 3, 1, batch_first=False)
        self.FC_output = torch.nn.Linear(self.dim_embedding_key * 3, 2)

        # map encoding (CNN)
        # scene: input shape (batch, classes, 180, 180)
        self.convScene_1 = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(4))
        self.convScene_2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(8))
        self.fc_featScene = nn.Linear(22*22*8, self.dim_embedding_key)

        self.fc_upsampling = nn.Linear(144, 1849)
        self.convScene_t1 = nn.Sequential(nn.ConvTranspose2d(1, 4, 3, stride=1), nn.ReLU(True))
        self.convScene_t2 = nn.Sequential(nn.ConvTranspose2d(4, 8, 3, stride=2,padding=1,output_padding=1), nn.ReLU(True))
        self.convScene_t3 = nn.Sequential(nn.ConvTranspose2d(8, 2, 3, stride=2,padding=1,output_padding=1))


        # self.convScene_1 = nn.Sequential(
        #     nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2))
        # self.convScene_2 = nn.Sequential(
        #     nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=2))
        # self.convScene_3 = nn.Sequential(
        #     nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=0),
        #     nn.ReLU())
        #
        # self.convScene_t1 = nn.Sequential(nn.ConvTranspose2d(4, 8, 4, stride=2), nn.ReLU(True))
        # self.convScene_t2 = nn.Sequential(nn.ConvTranspose2d(8, 16, 8, stride=4), nn.ReLU(True))
        # self.convScene_t3 = nn.Sequential(nn.ConvTranspose2d(16, 4, 8, stride=4), nn.ReLU())
        #


        #merge future and scene feature
        # self.fc_mapFuture = nn.Linear(self.dim_embedding_key*2, self.dim_embedding_key)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.maxpool = nn.MaxPool2d(2)

        # weight initialization: xavier
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.conv_past.weight)
        nn.init.kaiming_normal_(self.conv_fut.weight)
        nn.init.kaiming_normal_(self.encoder_past.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_past.weight_hh_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_ih_l0)
        nn.init.kaiming_normal_(self.encoder_fut.weight_hh_l0)
        nn.init.kaiming_normal_(self.decoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.decoder.weight_hh_l0)
        nn.init.kaiming_normal_(self.FC_output.weight)
        nn.init.kaiming_normal_(self.convScene_1[0].weight)
        nn.init.kaiming_normal_(self.convScene_2[0].weight)
        # nn.init.kaiming_normal_(self.fc_mapFuture.weight)

        nn.init.zeros_(self.conv_past.bias)
        nn.init.zeros_(self.conv_fut.bias)
        nn.init.zeros_(self.encoder_past.bias_ih_l0)
        nn.init.zeros_(self.encoder_past.bias_hh_l0)
        nn.init.zeros_(self.encoder_fut.bias_ih_l0)
        nn.init.zeros_(self.encoder_fut.bias_hh_l0)
        nn.init.zeros_(self.decoder.bias_ih_l0)
        nn.init.zeros_(self.decoder.bias_hh_l0)
        nn.init.zeros_(self.FC_output.bias)
        nn.init.zeros_(self.convScene_1[0].bias)
        nn.init.zeros_(self.convScene_2[0].bias)
        # nn.init.zeros_(self.fc_mapFuture.bias)

    def forward(self, past, future, scene):
        """
        Forward pass that encodes past and future and decodes the future.
        :param past: past trajectory
        :param future: future trajectory
        :return: decoded future
        """

        dim_batch = past.size()[0]
        zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 3).cuda()
        prediction = torch.Tensor().cuda()
        present = past[:, -1, :2].unsqueeze(1)

        # temporal encoding for past
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)

        # temporal encoding for future
        future = torch.transpose(future, 1, 2)
        future_embed = self.relu(self.conv_fut(future))
        future_embed = torch.transpose(future_embed, 1, 2)

        # sequence encoding
        output_past, state_past = self.encoder_past(story_embed)
        output_fut, state_fut = self.encoder_fut(future_embed)

        # map encoding
        # scene encoding
        #scene = scene.unsqueeze(3).permute(0, 3, 1, 2)
        scene_1 = self.convScene_1(scene) # 4,90,90
        scene_2 = self.maxpool(self.convScene_2(scene_1)) # 8,45,45 -> 8,22,22
        feat_scene = self.fc_featScene(scene_2.view(-1, 22*22*8)) # 48

        # concatenate
        state_conc = torch.cat((state_past, feat_scene.unsqueeze(0), state_fut), 2)

        #prediction
        input_fut = state_conc
        state_fut = zero_padding
        for i in range(self.future_len):
            output_decoder, state_fut = self.decoder(input_fut, state_fut)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction = torch.cat((prediction, coords_next), 1)
            present = coords_next
            input_fut = zero_padding

        #map
        feat_up = self.fc_upsampling(state_conc).view(-1,43,43).unsqueeze(1)
        scene_t1 = self.convScene_t1(feat_up)
        scene_t2 = self.convScene_t2(scene_t1)
        scene_t3 = self.convScene_t3(scene_t2)

        return prediction, scene_t3