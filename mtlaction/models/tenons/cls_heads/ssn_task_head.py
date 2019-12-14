import torch
import torch.nn as nn
import torch.nn.functional as F
from ...registry import HEADS

@HEADS.register_module
class SSNTaskHead(nn.Module):
    """SSN's task classification head"""

    def __init__(self,
                 join,
                 in_channels_task=100,
                 num_tasks=20,
                 dropout_ratio=0.8,
		 init_std=0.001):
    
        super(SSNTaskHead, self).__init__()

        self.join = join
        self.in_channels_task = in_channels_task
        self.num_tasks = num_tasks
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        # TODO: Add another intermidiate layer in the case of join != score
        assert join in ['score', 'act_feat', 'comp_feat'] # The branching point of the task branch

        # Single layer NN for prediction task
        self.task_fc = nn.Linear(in_channels_task, num_tasks)

    def init_weights(self):
        nn.init.normal_(self.task_fc.weight, 0, self.init_std)
        nn.init.constant_(self.task_fc.bias, 0)

    def prepare_test_fc(self, stpp_feat_multiplier):
        raise NotImplementedError
        # TODO: support the case of standalone=False
        self.test_fc = nn.Linear(self.activity_fc.in_features,
                                 self.activity_fc.out_features
                                 + self.completeness_fc.out_features * stpp_feat_multiplier
                                 + (self.regressor_fc.out_features * stpp_feat_multiplier if self.with_reg else 0))
        reorg_comp_weight = self.completeness_fc.weight.data.view(
                self.completeness_fc.out_features, stpp_feat_multiplier,
                self.activity_fc.in_features).transpose(0, 1).contiguous().view(-1, self.activity_fc.in_features)
        reorg_comp_bias = self.completeness_fc.bias.data.view(1, -1).expand(
                stpp_feat_multiplier, self.completeness_fc.out_features).contiguous().view(-1) / stpp_feat_multiplier

        weight = torch.cat((self.activity_fc.weight.data, reorg_comp_weight))
        bias = torch.cat((self.activity_fc.bias.data, reorg_comp_bias))

        if self.with_reg:
            reorg_reg_weight = self.regressor_fc.weight.data.view(
                    self.regressor_fc.out_features, stpp_feat_multiplier,
                    self.activity_fc.in_features).transpose(0, 1).contiguous().view(-1, self.activity_fc.in_features)
            reorg_reg_bias = self.regressor_fc.bias.data.view(1, -1).expand(
                    stpp_feat_multiplier, self.regressor_fc.out_features).contiguous().view(-1) / stpp_feat_multiplier
            weight = torch.cat((weight, reorg_reg_weight))
            bias = torch.cat((bias, reorg_reg_bias))

        self.test_fc.weight.data = weight
        self.test_fc.bias.data = bias
        return True


    def forward(self, input, test_mode=False):
        if not test_mode:
            task_feat = input
            if self.dropout is not None:
                task_feat = self.dropout(task_feat)

            task_score = self.task_fc(task_feat)

            return task_score
        else:
            raise NotImplementedError ("How could this work!!!")
            test_score = self.test_fc(input)
            return test_score

    def loss(self,
             task_score,
             task_labels,
             train_cfg):
        losses = dict()
        losses['loss_task'] = F.cross_entropy(task_score, task_labels) * train_cfg.ssn.loss_weight.task_loss_weight

        return losses
