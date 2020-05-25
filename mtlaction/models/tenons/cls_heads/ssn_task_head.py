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
