import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import HEADS

@HEADS.register_module
class SSNTaskHead(nn.Module):
    """
    Task head operating over pooled step scores
    """

    def __init__(self, num_steps, num_tasks, middle_layers, init_std=0.001):
        super(SSNTaskHead, self).__init__()

        self.in_features = num_steps
        self.out_features = num_tasks
        self.layers = [num_steps] + middle_layers + [num_tasks]
        self.init_std = init_std

        self.fcs = nn.ModuleList([])
        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fcs.append(nn.Linear(in_size, out_size))

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, self.init_std)
            nn.init.constant_(fc.bias, 0)

    def forward(self, step_scores):
        scores = step_scores
        for fc in self.fcs[:-1]:
            scores = F.relu(fc(scores))
        scores = self.fcs[-1](scores)
        return scores

    def loss(self, task_score, task_labels, train_cfg):
        losses = dict()
        losses['loss_task'] = F.cross_entropy(task_score, task_labels) * train_cfg.ssn.loss_weight.task_loss_weight
        return losses


@HEADS.register_module
class SSNAuxTaskHead(nn.Module):
    """
    Task head operating over the latent features Z
    """

    def __init__(self, in_feature_dim, num_tasks, middle_layers, init_std=0.001):
        super(SSNAuxTaskHead, self).__init__()

        self.in_features = in_feature_dim
        self.out_features = num_tasks
        self.layers = [in_feature_dim] + middle_layers + [num_tasks]
        self.init_std = init_std

        self.fcs = nn.ModuleList([])
        for in_size, out_size in zip(self.layers[:-1], self.layers[1:]):
            self.fcs.append(nn.Linear(in_size, out_size))

    def init_weights(self):
        for fc in self.fcs:
            nn.init.normal_(fc.weight, 0, self.init_std)
            nn.init.constant_(fc.bias, 0)

    def forward(self, feat):
        scores = feat
        for fc in self.fcs[:-1]:
            scores = F.relu(fc(scores))
        scores = self.fcs[-1](scores)
        return scores

    def loss(self, task_score, task_labels, train_cfg):
        losses = dict()
        losses['loss_aux_task'] = F.cross_entropy(task_score, task_labels) * train_cfg.ssn.loss_weight.aux_task_loss_weight
        return losses


class SSNTaskHeadOld(nn.Module):
    """SSN's task classification head"""

    def __init__(self,
                 join,
                 in_channels_task=100,
                 middle_layer=None,
                 num_tasks=20,
                 dropout_ratio=0.8,
		 init_std=0.001):
    
        super(SSNTaskHead, self).__init__()

        self.join = join
        self.in_channels_task = in_channels_task
        self.num_tasks = num_tasks
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.middle_layer = middle_layer

        assert middle_layer is None or isinstance(middle_layer, int)

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # TODO: Add another intermidiate layer in the case of join != score
        assert join in ['score', 'act_feat', 'comp_feat'] # The branching point of the task branch

        # Single layer NN for prediction task
        if self.middle_layer is not None:
            print ("Using Task Head with middle layer")
            self.task_fc1 = nn.Linear(in_channels_task, middle_layer)
            self.task_fc2 = nn.Linear(middle_layer, num_tasks)
        else:
            print ("Using Task Head without middle layer")
            self.task_fc = nn.Linear(in_channels_task, num_tasks)

    def init_weights(self):
        if self.middle_layer is not None:
            nn.init.normal_(self.task_fc1.weight, 0, self.init_std)
            nn.init.normal_(self.task_fc2.weight, 0, self.init_std)
            nn.init.constant_(self.task_fc1.bias, 0)
            nn.init.constant_(self.task_fc2.bias, 0)
        else:
            nn.init.normal_(self.task_fc.weight, 0, self.init_std)
            nn.init.constant_(self.task_fc.bias, 0)

    def forward(self, input):
        task_feat = input
        if self.dropout is not None:
            task_feat = self.dropout(task_feat)

        if self.middle_layer is not None:
            task_latent_feat = self.task_fc1(task_feat)
            task_score = self.task_fc2(task_latent_feat)
        else:
            task_score = self.task_fc(task_feat)

        return task_score

    def loss(self,
             task_score,
             task_labels,
             train_cfg):
        losses = dict()
        losses['loss_task'] = F.cross_entropy(task_score, task_labels) * train_cfg.ssn.loss_weight.task_loss_weight

        return losses
