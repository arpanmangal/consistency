import torch
import torch.nn as nn
import torch.nn.functional as F
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

                in_channels_task = cls_head.num_classes
                if self.task_join == 'act_feat': in_channels_task = cls_head.in_channels_activity
                elif self.task_join == 'comp_feat': in_channels_task = cls_head.in_channels_complete
                
                task_head.update({'in_channels_task': in_channels_task})
                del task_head['pooling']

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
        # print ('in SSN forward train')
        # print (kwargs.keys())
        # print (prop_scaling.shape)
        # print (prop_labels.shape)
        # print (reg_targets.shape)
        # print (task_labels.shape)

        assert num_modalities == 1
        img_group = kwargs['img_group_0']
        num_videos = img_group.shape[0]

        assert self.in_channels == img_group.shape[3] == 3
        # print ('hoho')
        # print (num_modalities)
        # print (img_group.shape)
        # print (self.in_channels)
        # print (img_group.shape)
        
        # img_group has a shape of [n, 8, 9, 3, 224, 224]
        img_group = img_group.reshape(
            (-1, self.in_channels) + img_group.shape[4:])

        # print (img_group.shape)
        x = self.extract_feat(img_group)
        # print (type(x), x.shape)
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        # print (type(x), x.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        # print (type(x), x.shape)
        activity_feat, completeness_feat = self.segmental_consensus(
            x, prop_scaling)
        # print (activity_feat.shape, completeness_feat.shape)
        losses = dict()
        if self.with_cls_head:
            activity_score, completeness_score, bbox_pred = self.cls_head(
                (activity_feat, completeness_feat))
            # print ('after cls_head')
            # print (activity_score.shape, completeness_score.shape, bbox_pred.shape)
            loss_cls = self.cls_head.loss(activity_score, completeness_score,
                                          bbox_pred, prop_type, prop_labels,
                                          reg_targets, self.train_cfg)
            # print (prop_type.shape, prop_labels.shape, reg_targets.shape, self.train_cfg)
            losses.update(loss_cls)

            if self.with_task_head and self.task_join == 'score':
                # Join the task branch over here
                # Step 1: Calculate the scores
                # print ('before task_head')
                s1 = F.softmax(activity_score[:, 1:], dim=1)
                s2 = torch.exp(completeness_score)
                # print (s1.shape)
                # print (s2.shape)
                combined_scores = s1 * s2
                # combined_scores = F.softmax(activity_score[:, 1:], dim=1) * torch.exp(completeness_score)
                # print (combined_scores.shape)
                combined_scores = combined_scores.reshape((num_videos, activity_score.shape[0] // num_videos, -1))
                # print ('yoho')
                # print (num_videos)
                # print (combined_scores.shape)
                # Step 2: Pool scores to create feature vector
                combined_scores = torch.mean(combined_scores, dim=1)
                # print (combined_scores.shape)

                # Step 3: Pass through NN and compute loss
                task_score = self.task_head(combined_scores)
                # print (task_score.squeeze().shape)
                # print (task_labels.shape)
                # print (prop_labels.shape)
                loss_task = self.task_head.loss(task_score, task_labels.squeeze(), self.train_cfg)
                # print (loss_task)
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
        assert num_modalities == 1
        img_group = kwargs['img_group_0']

        img_group = img_group[0]
        num_crop = img_group.shape[0]
        img_group = img_group.reshape(
            (num_crop, -1, self.in_channels) + img_group.shape[3:])
        num_ticks = img_group.shape[1]

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

        return rel_prop_list.cpu().numpy(), activity_scores.cpu().numpy(), \
            completeness_scores.cpu().numpy(), bbox_preds.cpu().numpy()
