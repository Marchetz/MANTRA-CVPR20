import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import pdb


class model_controllerMem_map(nn.Module):
    """
    Memory Network model with learnable writing controller.
    """

    def __init__(self, settings, model_pretrained):
        super(model_controllerMem_map, self).__init__()
        self.name_model = 'model_argo_map'

        # parameters
        self.use_cuda = settings["use_cuda"]
        self.dim_embedding_key = settings["dim_embedding_key"]
        self.num_prediction = settings["num_prediction"]
        self.past_len = settings["past_len"]
        self.future_len = settings["future_len"]
        self.th_memory = settings["th_memory"]

        # similarity criterion
        self.weight_read_past = []
        self.weight_read_scene = []
        self.index_max = []
        self.similarity = nn.CosineSimilarity(dim=1)

        # Memory
        self.memory_past = torch.Tensor().cuda()
        self.memory_scene = torch.Tensor().cuda()
        self.memory_fut = torch.Tensor().cuda()
        self.memory_count = []
        self.reading = []


        # layers
        self.conv_past = model_pretrained.conv_past
        self.conv_fut = model_pretrained.conv_fut

        self.encoder_past = model_pretrained.encoder_past
        self.encoder_fut = model_pretrained.encoder_fut
        self.decoder = model_pretrained.decoder
        self.FC_output = model_pretrained.FC_output

        #scene
        self.convScene_1 = model_pretrained.convScene_1
        self.convScene_2 = model_pretrained.convScene_2
        self.fc_featScene = model_pretrained.fc_featScene



        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        self.linear_controller = nn.Linear(1,1)
        self.memory_selector = nn.Linear(2,1)

        self.maxpool = nn.MaxPool2d(2)

        # weight initialization: xavier
        self.reset_parameters()

    def reset_parameters(self):

        nn.init.kaiming_normal_(self.linear_controller.weight)
        nn.init.zeros_(self.linear_controller.bias)


    def init_memory(self, data_train):
        """
        Initialization: write element in memory.
        :param data_train: dataset
        :return: None
        """

        self.memory_past = torch.Tensor().cuda()
        self.memory_fut = torch.Tensor().cuda()
        self.memory_scene = torch.Tensor().cuda()
        for i in range(1):

            j = random.randint(0, len(data_train)-1)
            past = data_train[j][1].unsqueeze(0)
            future = data_train[j][2].unsqueeze(0)
            scene = data_train[j][5]

            past = past.cuda()
            future = future.cuda()
            scene_input = torch.Tensor(np.eye(2, dtype=np.float32)[scene]).unsqueeze(0).permute(0, 3, 1, 2)
            scene_input = scene_input.cuda()

            # past encoding
            past = torch.transpose(past, 1, 2)
            story_embed = self.relu(self.conv_past(past))
            story_embed = torch.transpose(story_embed, 1, 2)
            output_past, state_past = self.encoder_past(story_embed)

            # future encoding
            future = torch.transpose(future, 1, 2)
            future_embed = self.relu(self.conv_fut(future))
            future_embed = torch.transpose(future_embed, 1, 2)
            output_fut, state_fut = self.encoder_fut(future_embed)

            # scene feature
            scene_1 = self.convScene_1(scene_input)  # 4,90,90
            scene_2 = self.maxpool(self.convScene_2(scene_1))  # 8,45,45 -> 8,22,22
            feat_scene = self.fc_featScene(scene_2.view(-1, 22 * 22 * 8))

            state_past = state_past.squeeze(0)
            state_fut = state_fut.squeeze(0)

            self.memory_past = torch.cat((self.memory_past, state_past), 0)
            self.memory_scene = torch.cat((self.memory_scene, feat_scene), 0)
            self.memory_fut = torch.cat((self.memory_fut, state_fut), 0)


    def check_memory(self, index):
        """
        method to generate a future track from past-future feature read from a index location of the memory.
        :param past: index of the memory
        :return: predicted future
        """

        mem_past_i = self.memory_past[index]
        mem_fut_i = self.memory_fut[index]
        zero_padding = torch.zeros(1, 1, 96).cuda()
        present = torch.zeros(1, 2).cuda()
        prediction_single = torch.Tensor().cuda()
        info_total = torch.cat((mem_past_i, mem_fut_i), 0)
        input_dec = info_total.unsqueeze(0).unsqueeze(0)
        state_dec = zero_padding
        for i in range(30):
            output_decoder, state_dec = self.decoder(input_dec, state_dec)
            displacement_next = self.FC_output(output_decoder)
            coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            prediction_single = torch.cat((prediction_single, coords_next), 1)
            present = coords_next
            input_dec = zero_padding
        return prediction_single

    def forward(self, past, scene, future=None, epoch=None, preds_scene = 10):
        """
        Forward pass.
        Train phase: training writing controller based on reconstruction error of the future.
        Test phase: Predicts future trajectory based on past trajectory and the future feature read from the memory.
        :param past: past trajectory
        :param future: future trajectory (in test phase)
        :return: predicted future (test phase), writing probability and tolerance rate (train phase)
        """
        #pdb.set_trace()
        dim_batch = past.size()[0]
        attention_save = torch.Tensor().cuda()

        prediction = torch.Tensor().cuda()
        present_temp = past[:, -1].unsqueeze(1)


        # past feature
        past = torch.transpose(past, 1, 2)
        story_embed = self.relu(self.conv_past(past))
        story_embed = torch.transpose(story_embed, 1, 2)
        output_past, state_past = self.encoder_past(story_embed)

        # scene feature
        scene_1 = self.convScene_1(scene) # 4,90,90
        scene_2 = self.maxpool(self.convScene_2(scene_1)) # 8,45,45 -> 8,22,22
        feat_scene = self.fc_featScene(scene_2.view(-1, 22*22*8)) # 48

        # Cosine similarity: past
        past_normalized = F.normalize(self.memory_past, p=2, dim=1)
        state_normalized = F.normalize(state_past.squeeze(), p=2, dim=1)
        self.weight_read_past = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1)

        # Cosine similarity: map
        sceneMem_normalized = F.normalize(self.memory_scene, p=2, dim=1)
        scene_normalized = F.normalize(feat_scene, p=2, dim=1)
        self.weight_read_scene = torch.matmul(sceneMem_normalized, scene_normalized.transpose(0, 1)).transpose(0, 1)

        #memory selector
        # reading = torch.Tensor().cuda()
        # attention = torch.cat((self.weight_read_past.unsqueeze(2), self.weight_read_scene.unsqueeze(2)), 2)
        # for a in range(attention.shape[1]):
        #     reading = torch.cat((reading, self.relu(self.memory_selector(attention[:,a])).unsqueeze(1)), 1)
        if future is None:
            attention = torch.cat((self.weight_read_past.unsqueeze(2), self.weight_read_scene.unsqueeze(2)), 2)
            sum_att = torch.mean(attention, dim=2)
            sum_att_sort = torch.sort(sum_att, descending=True)
            #index_att = sum_att_sort[1][:, :self.num_prediction]
            #index_att = torch.sort(self.weight_read_past, descending=True)[1][:, :self.num_prediction]

            att = torch.topk(self.weight_read_past, self.num_prediction)
            index_att = att[1]
            weight_att = att[0]


            zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 3).cuda()
            for i_track in range(preds_scene):
                present = present_temp
                prediction_single = torch.Tensor().cuda()
                ind = index_att[:, i_track]
                weight_ind = weight_att[:, i_track]

                scene_att = self.weight_read_scene[torch.arange(dim_batch), ind].unsqueeze(1)
                attention_save = torch.cat((attention_save, torch.cat((weight_ind.unsqueeze(1), scene_att), 1).unsqueeze(1)),1)

                info_future = self.memory_fut[ind]
                info_total = torch.cat((state_past, feat_scene.unsqueeze(0), info_future.unsqueeze(0)), 2)
                input_dec = info_total
                state_dec = zero_padding
                for i in range(self.future_len):
                    output_decoder, state_dec = self.decoder(input_dec, state_dec)
                    displacement_next = self.FC_output(output_decoder)
                    coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
                    prediction_single = torch.cat((prediction_single, coords_next), 1)

                    present = coords_next
                    input_dec = zero_padding
                prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)

        else:
            present_temp = torch.repeat_interleave(present_temp, len(self.memory_past), dim=0)
            state_past_repeat = torch.repeat_interleave(state_past,len(self.memory_past),dim=1)
            state_scene_repeat = torch.repeat_interleave(feat_scene.unsqueeze(0),len(self.memory_past),dim=1)
            info_future = self.memory_fut.unsqueeze(0).repeat(1, dim_batch, 1)

            info_total = torch.cat((state_past_repeat, state_scene_repeat, info_future), 2)
            zero_padding = torch.zeros(1, info_total.shape[1], self.dim_embedding_key * 3).cuda()
            present = present_temp
            input_dec = info_total
            state_dec = zero_padding
            prediction_single = torch.Tensor().cuda()
            for i in range(self.future_len):
                output_decoder, state_dec = self.decoder(input_dec, state_dec)
                displacement_next = self.FC_output(output_decoder)
                coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
                prediction_single = torch.cat((prediction_single, coords_next), 1)
                present = coords_next
                input_dec = zero_padding
            #prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)
            #pdb.set_trace()
            prediction = prediction_single.view(dim_batch, int(prediction_single.shape[0] / dim_batch), 30, 2)

        # for i_track in range(len(self.memory_past)):
        #     present = present_temp
        #     prediction_single = torch.Tensor().cuda()
        #     info_future = self.memory_fut[i_track].unsqueeze(0).unsqueeze(1).repeat(1, state_past.shape[1], 1)
        #     info_total = torch.cat((state_past, feat_scene.unsqueeze(0), info_future), 2)
        #     input_dec = info_total
        #     state_dec = zero_padding
        #     pdb.set_trace()
        #     for i in range(self.future_len):
        #         output_decoder, state_dec = self.decoder(input_dec, state_dec)
        #         displacement_next = self.FC_output(output_decoder)
        #         coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
        #         prediction_single = torch.cat((prediction_single, coords_next), 1)
        #         present = coords_next
        #         input_dec = zero_padding
        #     prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)


        if future is not None:
            tolerance = 0
            future_rep = future.unsqueeze(1).repeat(1, len(self.memory_past), 1, 1)
            distances = torch.norm(prediction - future_rep, dim=3)
            for step in range(future.shape[1]):
                tolerance += distances[:, :, step] < 0.07 * (step + 1)
            # tolerance_1s = torch.sum(distances[:, :, :10] < 0.5, dim=2)
            # tolerance_2s = torch.sum(distances[:, :, 10:20] < 1.0, dim=2)
            # tolerance_3s = torch.sum(distances[:, :, 20:30] < 2, dim=2)
            # #tolerance_4s = torch.sum(distances[:, :, 30:40] < 2, dim=2)
            # tolerance = tolerance_1s + tolerance_2s + tolerance_3s
            tolerance_rate = torch.max(tolerance, dim=1)[0].type(torch.FloatTensor) / future.shape[1]
            tolerance_rate = tolerance_rate.unsqueeze(1).cuda()

            # controller
            writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

            # future encoding
            future = torch.transpose(future, 1, 2)
            future_embed = self.relu(self.conv_fut(future))
            future_embed = torch.transpose(future_embed, 1, 2)
            output_fut, state_fut = self.encoder_fut(future_embed)

            index_writing = np.where(writing_prob.cpu() > 0.5)[0]
            past_to_write = state_past.squeeze()[index_writing]
            future_to_write = state_fut.squeeze()[index_writing]
            scene_to_write = feat_scene[index_writing]

            self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
            self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)
            self.memory_scene = torch.cat((self.memory_scene, scene_to_write), 0)

        else:
            return prediction, attention_save
        #pdb.set_trace()
        #pred_value = distances[:, :, 29].unsqueeze(2)
        return writing_prob, tolerance_rate

    def write_in_memory(self, past, future, scene):
            #TODO: change docs
            """
            Forward pass. Predicts future trajectory based on past trajectory and surrounding scene.
            :param past: past trajectory
            :param future: future trajectory
            :return: predicted future
            """

            dim_batch = past.size()[0]
            prediction = torch.Tensor().cuda()
            present_temp = past[:, -1].unsqueeze(1)
            # present_temp = torch.repeat_interleave(present_temp, len(self.memory_past), dim=0)

            # past temporal encoding
            past = torch.transpose(past, 1, 2)
            story_embed = self.relu(self.conv_past(past))
            story_embed = torch.transpose(story_embed, 1, 2)
            output_past, state_past = self.encoder_past(story_embed)

            # scene feature
            scene_1 = self.convScene_1(scene)  # 4,90,90
            scene_2 = self.maxpool(self.convScene_2(scene_1))  # 8,45,45 -> 8,22,22
            feat_scene = self.fc_featScene(scene_2.view(-1, 22 * 22 * 8))  # 48

            # Cosine similarity: past
            past_normalized = F.normalize(self.memory_past, p=2, dim=1)
            state_normalized = F.normalize(state_past.squeeze(), p=2, dim=1)
            self.weight_read_past = torch.matmul(past_normalized, state_normalized.transpose(0, 1)).transpose(0, 1)

            # Cosine similarity: map
            sceneMem_normalized = F.normalize(self.memory_scene, p=2, dim=1)
            scene_normalized = F.normalize(feat_scene, p=2, dim=1)
            self.weight_read_scene = torch.matmul(sceneMem_normalized, scene_normalized.transpose(0, 1)).transpose(0, 1)

            attention = torch.cat((self.weight_read_past.unsqueeze(2), self.weight_read_scene.unsqueeze(2)), 2)
            sum_att = torch.mean(attention, dim=2)
            sum_att_sort = torch.sort(sum_att, descending=True)
            #pdb.set_trace()
            #index_att = sum_att_sort[1][:, :self.num_prediction]
            index_att = torch.sort(self.weight_read_past, descending=True)[1][:,:self.num_prediction]


            if (len(self.memory_past) < self.num_prediction):
                num_prediction = len(self.memory_past)
            else:
                num_prediction = self.num_prediction

            zero_padding = torch.zeros(1, dim_batch, self.dim_embedding_key * 3).cuda()
            for i_track in range(num_prediction):
                present = present_temp
                prediction_single = torch.Tensor().cuda()
                ind = index_att[:, i_track]
                info_future = self.memory_fut[ind]
                info_total = torch.cat((state_past, feat_scene.unsqueeze(0), info_future.unsqueeze(0)), 2)
                input_dec = info_total
                state_dec = zero_padding
                for i in range(self.future_len):
                    output_decoder, state_dec = self.decoder(input_dec, state_dec)
                    displacement_next = self.FC_output(output_decoder)
                    coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
                    prediction_single = torch.cat((prediction_single, coords_next), 1)

                    present = coords_next
                    input_dec = zero_padding
                prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)

            # state_past_repeat = torch.repeat_interleave(state_past, len(self.memory_past), dim=1)
            # state_scene_repeat = torch.repeat_interleave(feat_scene.unsqueeze(0), len(self.memory_past), dim=1)
            # info_future = self.memory_fut.unsqueeze(0).repeat(1, dim_batch, 1)
            # info_total = torch.cat((state_past_repeat, state_scene_repeat, info_future), 2)
            # zero_padding = torch.zeros(1, info_total.shape[1], self.dim_embedding_key * 3).cuda()
            # present = present_temp
            # input_dec = info_total
            # state_dec = zero_padding
            # prediction_single = torch.Tensor().cuda()
            # for i in range(self.future_len):
            #     output_decoder, state_dec = self.decoder(input_dec, state_dec)
            #     displacement_next = self.FC_output(output_decoder)
            #     coords_next = present + displacement_next.squeeze(0).unsqueeze(1)
            #     prediction_single = torch.cat((prediction_single, coords_next), 1)
            #     present = coords_next
            #     input_dec = zero_padding
            # # prediction = torch.cat((prediction, prediction_single.unsqueeze(1)), 1)
            # # pdb.set_trace()
            # prediction = prediction_single.view(dim_batch, int(prediction_single.shape[0] / dim_batch), 30, 2)


            #TODO: change tolerance rate
            tolerance = 0
            future_rep = future.unsqueeze(1).repeat(1, num_prediction, 1, 1)
            distances = torch.norm(prediction - future_rep, dim=3)
            for step in range(future.shape[1]):
                tolerance += distances[:, :, step] < 0.07 * (step + 1)
            tolerance_rate = torch.max(tolerance, dim=1)[0].type(torch.FloatTensor) / future.shape[1]
            tolerance_rate = tolerance_rate.unsqueeze(1).cuda()

            #controller
            writing_prob = torch.sigmoid(self.linear_controller(tolerance_rate))

            # future encoding
            future = torch.transpose(future, 1, 2)
            future_embed = self.relu(self.conv_fut(future))
            future_embed = torch.transpose(future_embed, 1, 2)
            output_fut, state_fut = self.encoder_fut(future_embed)

            index_writing = np.where( writing_prob.cpu() > 0.5)[0]
            past_to_write = state_past.squeeze()[index_writing]
            future_to_write = state_fut.squeeze()[index_writing]
            scene_to_write = feat_scene[index_writing]

            self.memory_past = torch.cat((self.memory_past, past_to_write), 0)
            self.memory_fut = torch.cat((self.memory_fut, future_to_write), 0)
            self.memory_scene = torch.cat((self.memory_scene, scene_to_write), 0)

