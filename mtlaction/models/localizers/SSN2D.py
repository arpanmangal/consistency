import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from .base import BaseLocalizer
from .. import builder
from ..registry import LOCALIZERS

@LOCALIZERS.register_module
class SSN2D(BaseLocalizer):

    def __init__(self,
                 backbone,
                 modality='RGB',
                 in_channels=3,
                 spatial_temporal_module=None,
                 dropout_ratio=0.5,
                 segmental_consensus=None,
                 cls_head=None,
                 task_head=None,
                 train_cfg=None,
                 test_cfg=None):

        super(SSN2D, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        self.modality = modality
        self.in_channels = in_channels

        if spatial_temporal_module is not None:
            self.spatial_temporal_module = (
                builder.build_spatial_temporal_module(spatial_temporal_module))
        else:
            raise NotImplementedError

        self.dropout_ratio = dropout_ratio
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)

        if segmental_consensus is not None:
            self.segmental_consensus = builder.build_segmental_consensus(
                segmental_consensus)
        else:
            raise NotImplementedError

        if cls_head is not None:
            self.cls_head = builder.build_head(cls_head)
            self.is_test_prepared = False
        else:
            raise NotImplementedError

        if task_head is not None:
            self.task_join = task_head.join # Place where to join the task head
            if self.task_join is not None:
                assert self.task_join in ['score', 'act_feat', 'comp_feat']
                assert task_head.pooling in ['mean', 'max']

                # Way to pool the features for task head
                self.task_feat_pooling = task_head.pooling

                if self.task_join != 'score': raise NotImplementedError #TODO
                if self.task_feat_pooling != 'mean': raise NotImplementedError # TODO - Probably not reqd.

                in_channels_task = cls_head.num_classes
                if self.task_join == 'act_feat': in_channels_task = cls_head.in_channels_activity
                elif self.task_join == 'comp_feat': in_channels_task = cls_head.in_channels_complete
                
                task_head.update({'num_steps': in_channels_task})
                del task_head['pooling']; del task_head['join']

                self.task_head = builder.build_head(task_head)
                print ("Something corresponding is missing here")
            else:
                self.task_head = None
        else:
            raise NotImplementedError

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert modality in ['RGB', 'Flow', 'RGBDiff']

        self.init_weights()

        if modality == 'Flow' or modality == 'RGBDiff':
            self._construct_2d_backbone_conv1(in_channels)

    @property
    def with_spatial_temporal_module(self):
        return (hasattr(self, 'spatial_temporal_module') and
                self.spatial_temporal_module is not None)

    @property
    def with_segmental_consensus(self):
        return (hasattr(self, 'segmental_consensus') and
                self.segmental_consensus is not None)

    @property
    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    @property
    def with_task_head(self):
        return hasattr(self, 'task_head') and self.task_head is not None

    def _construct_2d_backbone_conv1(self, in_channels):
        modules = list(self.backbone.modules())
        first_conv_idx = list(filter(lambda x: isinstance(
            modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (in_channels, ) + kernel_size[2:]
        new_kernel_data = params[0].data.mean(dim=1, keepdim=True).expand(
            new_kernel_size).contiguous()  # make contiguous!

        new_conv_layer = nn.Conv2d(in_channels, conv_layer.out_channels,
                                   conv_layer.kernel_size,
                                   conv_layer.stride, conv_layer.padding,
                                   bias=True if len(params) == 2 else False)
        new_conv_layer.weight.data = new_kernel_data
        if len(params) == 2:
            new_conv_layer.bias.data = params[1].data
        # remove ".weight" suffix to get the layer layer_name
        layer_name = list(container.state_dict().keys())[0][:-7]
        setattr(container, layer_name, new_conv_layer)

    def init_weights(self):
        super(SSN2D, self).init_weights()
        self.backbone.init_weights()

        if self.with_spatial_temporal_module:
            self.spatial_temporal_module.init_weights()

        if self.with_segmental_consensus:
            self.segmental_consensus.init_weights()

        if self.with_cls_head:
            self.cls_head.init_weights()

        if self.with_task_head:
            self.task_head.init_weights()

    def extract_feat(self, img_group):
        x = self.backbone(img_group)
        return x

    def forward_train(self,
                      num_modalities,
                      img_meta,
                      prop_scaling,
                      prop_type,
                      prop_labels,
                      reg_targets,
                      task_labels,
                      **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']
        num_videos = img_group.shape[0]

        assert self.in_channels == img_group.shape[3] == 3
        
        # img_group has a shape of [n, 8, 9, 3, 224, 224]
        # n is 2 for us
        # after below the shape becomes [144, 3, 244, 244]
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[4:])

        # after below x.shape == [144, 1024, 7, 7]
        x = self.extract_feat(img_group)

        # after below x.shape == [144, 1024, 1, 1]
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)

        # after below x.shape == [144, 1024, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        
        # below shapes [16, 1024], [16, 3072]
        activity_feat, completeness_feat = self.segmental_consensus(
            x, prop_scaling)
        
        losses = dict()
        if self.with_cls_head:
            # shapes = [16, 32], [16, 31], [16, 62]
            activity_score, completeness_score, bbox_pred = self.cls_head(
                (activity_feat, completeness_feat))
            loss_cls = self.cls_head.loss(activity_score, completeness_score,
                                          bbox_pred, prop_type, prop_labels,
                                          reg_targets, self.train_cfg)
            losses.update(loss_cls)

            if self.with_task_head and self.task_join == 'score':
                # Join the task branch over here
                # Step 1: Calculate the scores
                s1 = F.softmax(activity_score[:, 1:], dim=1)
                s2 = torch.exp(completeness_score)
                combined_scores = s1 * s2
                combined_scores = combined_scores.reshape((num_videos, activity_score.shape[0] // num_videos, -1))
                
                # Step 2: Pool scores to create feature vector
                if self.task_feat_pooling == 'mean':
                    combined_scores = torch.mean(combined_scores, dim=1)
                else:
                    combined_scores = torch.max(combined_scores, dim=1).values

                # Step 3: Pass through NN and compute loss
                # shape == [2, 7]
                task_score = self.task_head(combined_scores)
                loss_task = self.task_head.loss(task_score, task_labels.squeeze(), self.train_cfg)
                losses.update(loss_task)

        return losses

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     rel_prop_list,
                     scaling_list,
                     prop_tick_list,
                     reg_stats,
                     **kwargs):
        # Call different function for perturbation
        if 'perturb' in self.test_cfg.ssn:
            return self.forward_test_perturb(num_modalities, img_meta, rel_prop_list, scaling_list, 
                                             prop_tick_list, reg_stats, **kwargs)

        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        img_group = img_group[0]
        num_crop = img_group.shape[0]
        
        assert self.in_channels == img_group.shape[2] == 3

        img_group = img_group.reshape(
            (num_crop, -1, self.in_channels) + img_group.shape[3:])
        num_ticks = img_group.shape[1] # Number of sample frames from total frames

        output = []
        minibatch_size = self.test_cfg.ssn.sampler.batch_size
        for ind in range(0, num_ticks, minibatch_size):
            chunk = img_group[:, ind:ind + minibatch_size, ...].view(
                (-1,) + img_group.shape[2:])
            x = self.extract_feat(chunk.cuda())
            x = self.spatial_temporal_module(x)
            # merge crop to save memory
            # TODO: A smarte way of dealing with arbitary long videos
            x = x.reshape((num_crop, x.size(0)//num_crop, -1)).mean(dim=0)
            output.append(x)
        output = torch.cat(output, dim=0)
        
        if not self.is_test_prepared:
            self.is_test_prepared = self.cls_head.prepare_test_fc(
                self.segmental_consensus.feat_multiplier)
        output = self.cls_head(output, test_mode=True)

        rel_prop_list = rel_prop_list.squeeze(0)
        prop_tick_list = prop_tick_list.squeeze(0)
        scaling_list = scaling_list.squeeze(0)
        reg_stats = reg_stats.squeeze(0)
        (activity_scores, completeness_scores,
         bbox_preds) = self.segmental_consensus(
            output, prop_tick_list, scaling_list)

        if bbox_preds is not None:
            bbox_preds = bbox_preds.view(-1, self.cls_head.num_classes, 2)
            bbox_preds[:, :, 0] = bbox_preds[:, :, 0] * \
                reg_stats[1, 0] + reg_stats[0, 0]
            bbox_preds[:, :, 1] = bbox_preds[:, :, 1] * \
                reg_stats[1, 1] + reg_stats[0, 1]

        if self.with_task_head and self.task_join == 'score':
            # Join the task branch over here
            # Step 1: Calculate the scores
            s1 = F.softmax(activity_scores[:, 1:], dim=1)
            s2 = torch.exp(completeness_scores)
            combined_scores = s1 * s2

            # Step 2: Pool scores to create feature vector
            if self.task_feat_pooling == 'mean':
                combined_scores = torch.mean(combined_scores, dim=0)
            else:
                combined_scores = torch.max(combined_scores, dim=0).values

            # Step 3: Pass through NN and compute loss
            task_score = self.task_head(combined_scores)
            task_score = task_score.cpu().numpy()
        else:
            task_score = None

        return rel_prop_list.cpu().numpy(), activity_scores.cpu().numpy(), \
            completeness_scores.cpu().numpy(), bbox_preds.cpu().numpy(), task_score

    def forward_test_perturb(self,
                     num_modalities,
                     img_meta,
                     rel_prop_list,
                     scaling_list,
                     prop_tick_list,
                     reg_stats,
                     **kwargs):

        # print ('perturbing !!!')
        assert self.test_cfg.ssn.perturb.type in ['oneway', 'twoway']
        assert self.with_task_head and self.task_join == 'score'

        # Preparing cls_head for test mode
        if not self.is_test_prepared:
            self.is_test_prepared = self.cls_head.prepare_test_fc(
                self.segmental_consensus.feat_multiplier)

        # Save the model so that we could restore the weights at the end
        tmp_model_file = 'tmp_models/model_' + str(np.random.random()) + '.tmp.pth'
        self._save_tmp_model(tmp_model_file)

        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        img_group = img_group[0]
        num_crop = img_group.shape[0]
        
        assert self.in_channels == img_group.shape[2] == 3

        # img_group.shape == [1, num_ticks, 3, 224, 224]
        img_group = img_group.reshape(
            (num_crop, -1, self.in_channels) + img_group.shape[3:])
        
        num_ticks = img_group.shape[1] # Number of sample frames from total frames

        def forward_pass_train (img_group, num_ticks, rel_prop_list, scaling_list, prop_tick_list, reg_stats):
            """
            The forward pass while training
            """
            x = self.extract_feat(img_group)
            x = self.spatial_temporal_module(x)
            x = self.dropout(x)
            activity_feat, completeness_feat = self.segmental_consensus(
                x, scaling_list.squeeze(0))
            activity_score, completeness_score, bbox_pred = self.cls_head(
                (activity_feat, completeness_feat))
            s1 = F.softmax(activity_score[:, 1:], dim=1)
            s2 = torch.exp(completeness_score)
            combined_scores = s1 * s2
            combined_scores = combined_scores.reshape((num_videos, activity_score.shape[0] // num_videos, -1))
            combined_scores = torch.mean(combined_scores, dim=1)
            task_score = self.task_head(combined_scores)
            return task_score

        def forward_pass (img_group, num_ticks, rel_prop_list, scaling_list, prop_tick_list, reg_stats):
            """
            The forward pass through the network
            """
            output = []
            minibatch_size = self.test_cfg.ssn.sampler.batch_size
            for ind in range(0, num_ticks, minibatch_size):
                chunk = img_group[:, ind:ind + minibatch_size, ...].view(
                    (-1,) + img_group.shape[2:]) # chunk.shape == [16, 3, 244, 244]
                x = self.extract_feat(chunk.cuda()) # x.shape == [16, 1024, 7, 7]
                x = self.spatial_temporal_module(x) # x.shape == [16, 1024, 1, 1]
                # merge crop to save memory
                # TODO: A smarte way of dealing with arbitary long videos
                x = x.reshape((num_crop, x.size(0)//num_crop, -1)).mean(dim=0) # x.shape == [16, 1024]
                output.append(x)
            output = torch.cat(output, dim=0) # output.shape == [num_ticks, 1024]
            
            output = self.cls_head(output, test_mode=True)  # output.shape == [num_ticks, 311], 311 = 32 + 3(31+62)

            # rel_prop_list1 = rel_prop_list
            rel_prop_list = rel_prop_list.squeeze(0)
            prop_tick_list = prop_tick_list.squeeze(0)
            scaling_list = scaling_list.squeeze(0)
            reg_stats = reg_stats.squeeze(0)
            # below shapes: [n, 32], [n, 31], [n, 62]
            (activity_scores, completeness_scores,
            bbox_preds) = self.segmental_consensus(
                output, prop_tick_list, scaling_list)

            if bbox_preds is not None:
                bbox_preds = bbox_preds.view(-1, self.cls_head.num_classes, 2)
                bbox_preds[:, :, 0] = bbox_preds[:, :, 0] * \
                    reg_stats[1, 0] + reg_stats[0, 0]
                bbox_preds[:, :, 1] = bbox_preds[:, :, 1] * \
                    reg_stats[1, 1] + reg_stats[0, 1]

            # Join the task branch over here
            # Step 1: Calculate the scores
            s1 = F.softmax(activity_scores[:, 1:], dim=1)
            s2 = torch.exp(completeness_scores)
            combined_scores = s1 * s2

            # Step 2: Pool scores to create feature vector
            if self.task_feat_pooling == 'mean':
                combined_scores = torch.mean(combined_scores, dim=0)
            else:
                combined_scores = torch.max(combined_scores, dim=0).values

            # Step 3: Pass through NN and compute loss
            task_score = self.task_head(combined_scores).unsqueeze(0) # task_score.shape == [1, 7]

            return rel_prop_list, activity_scores, completeness_scores, bbox_preds, task_score

        # Processing all the imgs leads to CUDA out of memory => So we will use lesser images
        def reduce_img_group_size(num_ticks):
            reduction_factor = (num_ticks // 100) + 1
            return list(range(0, num_ticks, reduction_factor))

        self.eval()
        print ('lr = ', self.test_cfg.ssn.perturb.optimizer['lr'])
        optimizer = optim.SGD(self.parameters(),
                              lr=self.test_cfg.ssn.perturb.optimizer['lr'],
                              momentum=self.test_cfg.ssn.perturb.optimizer['momentum'])
        # criterion = nn.CrossEntropyLoss()

        img_group_short = img_group[:, reduce_img_group_size(num_ticks), ...]

        print ('at the top')
        print (img_group.shape, img_group_short.shape) # [1, 149, 3, 224, 224]
        # print (type(rel_prop_list), rel_prop_list.shape) # [1, 35, 2]
        # print (type(scaling_list), scaling_list.shape) # [1, 35, 2]
        # print (type(prop_tick_list), prop_tick_list.shape) #[1, 35, 4])
        # print (type(reg_stats), reg_stats.shape) # [1, 2, 2]
        # print (type(img_group_short), img_group_short.shape)
        # print ('-------------------------------------------')
        # print (rel_prop_list, rel_prop_list.dtype)
        # print ('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')
        # print (prop_tick_list, prop_tick_list.dtype)
        # print ('===========================================')
        # print (scaling_list, scaling_list.dtype)
        # print ('******************************************')
        # exit(0)

        # Update weights
        num_times = 3
        for _ in range(num_times):
            # with torch.no_grad():
            with torch.enable_grad():
                # Early stopping
                optimizer.zero_grad()

                # Forward pass
                _, _, _, _, task_score = forward_pass(img_group_short.clone().detach(), img_group_short.shape[1],
                                                      rel_prop_list.clone().detach(), scaling_list.clone().detach(),
                                                      prop_tick_list.clone().detach(), reg_stats.clone().detach())
                task_predictions = torch.argmax(task_score).unsqueeze(0)

                # print ('in the loop')
                # print (type(rel_prop_list), rel_prop_list.shape)
                # print (type(scaling_list), scaling_list.shape)
                # print (type(prop_tick_list), prop_tick_list.shape)
                # print (type(reg_stats), reg_stats.shape)
                # print (type(img_group_short), img_group_short.shape)
                # print ('-------------------------------------------')

                # Backprop
                # loss = criterion(task_score, hard_labels)
                print (task_score, task_predictions)
                loss = F.cross_entropy(task_score, task_predictions)
                loss.backward()
                # optimizer.step()
                # print ('Task Loss', loss.item())
        print ('---------------------------------------------------\n')

        # Final forward pass
        # Restoring weights to confirm they are not changed
        # self._restore_model(tmp_model_file)
        # self.eval()
        with torch.no_grad():
            rel_prop_list, activity_scores, completeness_scores, bbox_preds,\
                task_score = forward_pass(img_group, num_ticks, rel_prop_list, scaling_list, prop_tick_list, reg_stats)
            print ('############')
            print (task_score)
                
        # Restore the model
        self._restore_model(tmp_model_file)

        return rel_prop_list.cpu().numpy(), activity_scores.cpu().numpy(), \
            completeness_scores.cpu().numpy(), bbox_preds.cpu().numpy(), task_score.cpu().numpy()

    def _save_tmp_model(self, tmp_model_file):
        """
        Save the model weights in a file so that it could be updated later
        """
        torch.save(self.state_dict(), tmp_model_file)

    def _restore_model(self, tmp_model_file):
        """
        Restore the model weights from the file
        """
        if torch.cuda.is_available():
            model = torch.load(tmp_model_file)
        else:
            model = torch.load(tmp_model_file, map_location=torch.device('cpu'))

        self.load_state_dict(model)