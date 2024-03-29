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
                 aux_task_head=None,
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

        # At train time: same as config file
        # At test time: {'type': 'STPPReorganized', 'standalong_classifier': True, 'feat_dim': 311,
        #            'act_score_len': 32, 'comp_score_len': 31, 'reg_score_len': 62, 'stpp_cfg': (1, 1, 1)}
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

        # Task head for MTL
        if task_head is not None:
            self.task_join = task_head.join # Place where to join the task head
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

            print (task_head)

            self.task_head = builder.build_head(task_head)
            print ("Something corresponding is missing here")
        else:
            self.task_head = None

        # Task middleware for MTL++
        if aux_task_head is not None:
            if not self.with_task_head:
                raise NotImplementedError("Task Head is reqd if using MTL++")
            
            # Prepare and initialize the head
            in_channels_task = cls_head.in_channels_activity
            num_tasks = aux_task_head.num_tasks
            assert task_head.num_tasks == num_tasks
            assert 'in_channels_tasks' in cls_head and cls_head.in_channels_tasks == num_tasks
            aux_task_head.update({'in_feature_dim': in_channels_task})
            self.aux_task_head = builder.build_head(aux_task_head)
        else:
            assert 'in_channels_tasks' not in cls_head
            self.aux_task_head = None

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        assert modality in ['RGB', 'Flow', 'RGBDiff']

        self.init_weights()

        if modality == 'Flow' or modality == 'RGBDiff':
            self._construct_2d_backbone_conv1(in_channels)

        # Freeze backbone weights if fine-tuning
        if 'freeze' in self.train_cfg.ssn:
            if 'backbone' in self.train_cfg.ssn.freeze:
                print ("Freezing the backbone weights")
                for param in self.backbone.parameters():
                    param.requires_grad = False

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
    def with_aux_task_head(self):
        return hasattr(self, 'aux_task_head') and self.aux_task_head is not None

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

        if self.with_aux_task_head:
            self.aux_task_head.init_weights()

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
   
        one_hot_task_input = None
        if self.with_aux_task_head:
            num_per_video = activity_feat.shape[0] // num_videos
            input_feat = activity_feat.reshape((num_videos, num_per_video, -1))
            input_feat = torch.mean(input_feat, dim=1)
            aux_task_pred = self.aux_task_head(input_feat)
            loss_aux_task = self.aux_task_head.loss(aux_task_pred, task_labels.squeeze(), self.train_cfg)
            losses.update(loss_aux_task)

            # While training let's use the actual task labels as the aux_task_pred to reduce the errors
            num_tasks = aux_task_pred.shape[1]
            one_hot_task_input = (task_labels == torch.arange(num_tasks).cuda().reshape(1, num_tasks)).float()
            one_hot_task_input = one_hot_task_input.repeat(1, num_per_video).view(-1, num_tasks)

        if self.with_cls_head:
            # shapes = [16, 32], [16, 31], [16, 62]
            activity_score, completeness_score, bbox_pred = self.cls_head(
                (activity_feat, completeness_feat, one_hot_task_input))
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

        # Empty cache
        torch.cuda.empty_cache()
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
        if self.with_aux_task_head and 'perturb' in self.test_cfg.ssn and self.test_cfg.ssn.perturb.numbps > 0:
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
                self.segmental_consensus.feat_multiplier,
                aux_task_head=self.with_aux_task_head)

        # if aux_task_head is available, we will use it to predict task from the [n, 1024 features itself]
        # next it will be passed as input to the cls_head
        aux_task_pred = None
        if self.with_aux_task_head:
            n_ticks = output.size(0)
            input_feat = torch.mean(output, dim=0).reshape(1, -1)
            aux_task_pred = self.aux_task_head(input_feat)

            num_tasks = aux_task_pred.size(1)
            aux_task_pred = aux_task_pred.repeat(1, n_ticks).view(-1, num_tasks)
            assert aux_task_pred.size(0) == output.size(0)

        # input: output.shape == [n, 1024]
        # output: output.shape == [n, 311]
        # n is variable -- num proposals for the given video
        output = self.cls_head((output, aux_task_pred), test_mode=True)

        # meaning of the below quantities
        # rel_prop_list = [n, 2] torch tensor, containing (s,e) tuples of proposal starting and ending times
        # 0 <= s < e <= 1
        # 
        # prop_list = [n, 4] torch tensor, purpose: UNKOWN
        # example: torch.Size([23, 4])
        #     tensor([[  0,   0, 130, 133],
        #             [  0,   6, 121, 133],
        #             [  0,  10, 112, 133],
        #             [  0,  10, 100, 133],
        #             [  0,  10,  89, 128],
        #             [  0,  10,  79, 114],
        #             [  0,  10,  59,  83],
        #             [ 34,  60, 112, 133],
        #             ...
        # 
        # scaling_list = [n, 2] torch tensor, purpose: UNKOWN
        # example: torch.Size([23, 2])
        #         tensor([[0.0000, 0.0437],
        #                 [0.1139, 0.2073],
        #                 [0.2102, 0.4007],
        #                 [0.2384, 0.7225],
        #                 [0.2723, 1.0000],
        #                 [0.3107, 1.0000],
        #                 [0.4414, 1.0000],
        #                 [1.0000, 0.7821],
        #                 [1.0000, 1.0000],
        #                 [1.0000, 1.0000],
        #                 [1.0000, 1.0000],
        #                 [1.0000, 1.0000],
        #                 [1.0000, 1.0000],
        #                 [1.0000, 1.0000],
        #                 [1.0000, 1.0000],
        # 
        # reg_stats: [2, 2] torch tensor, pupose: UNKOWN
        # example: torch.Size([2, 2])
        #     tensor([[-0.0032, -0.0267],
        #             [ 0.0960,  0.1922]]
        # 

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

        if self.aux_task_head:
            # do a backprop on KL loss bw the two predicted tasks
            # TODO
            bp = 'Done'

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
        # assert self.test_cfg.ssn.perturb.type in ['oneway', 'twoway']
        assert self.with_task_head and self.task_join == 'score'
        assert self.with_aux_task_head

        # Preparing cls_head for test mode, preparing early so as to save
        if not self.is_test_prepared:
            self.is_test_prepared = self.cls_head.prepare_test_fc(
                self.segmental_consensus.feat_multiplier,
                aux_task_head=self.with_aux_task_head)

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
            
            # we will use aux_task_head to predict task from the [n, 1024 features itself]
            # next it will be passed as input to the cls_head
            n_ticks = output.size(0)
            input_feat = torch.mean(output, dim=0).reshape(1, -1)
            aux_task_pred = self.aux_task_head(input_feat)

            num_tasks = aux_task_pred.size(1)
            aux_task_pred = aux_task_pred.repeat(1, n_ticks).view(-1, num_tasks)
            assert aux_task_pred.size(0) == output.size(0)

            # input: output.shape == [n, 1024]
            # output: output.shape == [n, 311]
            # n is variable -- num proposals for the given video
            # 311 = 32 + 3(31+62)
            output = self.cls_head((output, aux_task_pred), test_mode=True)

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
                mean_combined_scores = torch.mean(combined_scores, dim=0)
            else:
                mean_combined_scores = torch.max(combined_scores, dim=0).values

            # Step 3: Pass through NN and compute loss
            task_score = self.task_head(mean_combined_scores).unsqueeze(0) # task_score.shape == [1, 7]

            return rel_prop_list, activity_scores, completeness_scores, bbox_preds, task_score, aux_task_pred

        # Update weights
        self.eval()
        # criterion = nn.CrossEntropyLoss()
        num_times = self.test_cfg.ssn.perturb.numbps
        for bp in range(num_times):
            # Early stopping
            print ('#%d / %d' % (bp, num_times))
            with torch.enable_grad():
                # Freeze heads and create optimizer
                for param in self.task_head.parameters():
                    param.requires_grad = False
                for param in self.aux_task_head.parameters():
                    param.requires_grad = False
                for param in self.cls_head.parameters():
                    param.requires_grad = False

                optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                lr=self.test_cfg.ssn.perturb.optimizer['lr'],
                                momentum=self.test_cfg.ssn.perturb.optimizer['momentum'])


                # Task -> Step updation
                # Forward pass
                # optimizer.zero_grad()
                # _, _, _, _, task_score, _ = forward_pass(img_group.clone().detach(), img_group.shape[1],
                #                                       rel_prop_list.clone().detach(), scaling_list.clone().detach(),
                #                                       prop_tick_list.clone().detach(), reg_stats.clone().detach())
                # task_predictions = torch.argmax(task_score).unsqueeze(0)

                # # Backprop
                # # loss = criterion(task_score, hard_labels)
                # print (task_score, task_predictions)
                # loss = F.cross_entropy(task_score, task_predictions)
                # loss.backward()
                # torch.cuda.empty_cache() # To empty the cache from previous iterations
                # optimizer.step()
                # print ('Task Loss', loss.item())

                # Task -> Step updation
                # Forward pass
                optimizer.zero_grad()
                _, _, _, _, task_score, aux_task_pred = forward_pass(img_group.clone().detach(), img_group.shape[1],
                                                      rel_prop_list.clone().detach(), scaling_list.clone().detach(),
                                                      prop_tick_list.clone().detach(), reg_stats.clone().detach())

                assert task_score.size(0) == 1 # Exactly one row (since one video)
                # task_score = torch.softmax(task_score, dim=1)
                # aux_task_pred = torch.softmax(aux_task_pred[:1], dim=1) 

                # print (task_score)
                # print (aux_task_pred)
                task_ground_truth_prob = torch.softmax(task_score, dim=1)
                task_pred = torch.softmax(aux_task_pred[:1], dim=1) # All rows are same

                # print (task_ground_truth_prob)
                # print (task_pred)
                # exit(0)

                # loss = F.cross_entropy(activity_scores, step_predictions)
                loss = F.kl_div(task_pred.log(), task_ground_truth_prob, None, None, 'sum')

                loss.backward()
                torch.cuda.empty_cache() # To empty the cache from previous iterations
                optimizer.step()
                print ('Step Loss', loss.item())
                # print (combined_scores.shape)
                # print (step_predictions.shape)
                # print (combined_scores)
                # print (torch.sum(combined_scores, dim=1))
                # print (torch.sum(torch.exp(combined_scores), dim=1))
                # exit(0)


        # Final forward pass
        # Restoring weights to confirm they are not changed
        # self._restore_model(tmp_model_file)
        # self.eval()
        with torch.no_grad():
            rel_prop_list, activity_scores, completeness_scores, bbox_preds,\
                task_score, _ = forward_pass(img_group, num_ticks, rel_prop_list, scaling_list, prop_tick_list, reg_stats)
            print ('############')
            print (task_score)
            torch.cuda.empty_cache()
        print ('---------------------------------------------------\n')
                
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