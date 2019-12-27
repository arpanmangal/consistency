"""
Networks for task head
Predict task_id from the step scores
"""

import os, glob
import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class NMS:
    """
    Contains functions for doing temporal nms
    """
    def __init__ (self, props, scores, regs):
        """
        props: V*N*2 array containing N proposals (s,e)
        scores: V*N*S array containing step scores for each proposal
        regs: V*N*S*2 array containing regression parameters for each proposal
        """
        assert len(props) == len(scores) == len(regs)
        for p, s, r in zip(props, scores, regs):
            N, S = s.shape
            assert p.shape == (N, 2)
            assert r.shape == (N, S, 2)
            
        self.props = props
        self.scores = scores
        self.regs = regs

    def _perform_regression(self):
        """
        Perform regression on the props using regs
        """
        reg_props = []
        for p, r in zip(self.props, self.regs):
            t0 = p[:, 0] # Start time
            t1 = p[:, 1] # End time
            center = (t0 + t1) / 2.0
            duration = t1 - t2

            new_center = center + duration * r[:, 0]
            new_duration = duration * np.exp(r[:, 1])

            new_props = np.concatenate((
                np.clip(new_center - new_duration / 2, 0, 1)[:, None],
                np.clip(new_center + new_duration / 2, 0, 1)[:, None]
                ), axis=1)
            
            assert new_props.shape == p.shape
            reg_props.append(new_props)
        
        return reg_props

    def _perform_temporal_nms(self, detections, thresh):
        t0 = detections[:, 0]
        t1 = detections[:, 1]
        scores = detections[:, 2]

        durations = t1 - t0
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            tt0 = np.maximum(t0[i], t0[order[1:]])
            tt1 = np.minimum(t1[i], t1[order[1:]])
            intersection = tt1 - tt0
            iou = intersection / \
                (durations[i] + durations[order[1:]] - intersection).astype(float)

            inds = np.where(iou <= thresh)[0]
            order = order[inds + 1]

        return keep
        # return detections[keep, :]

    def temporal_nms_gola(self, thres, no_reg=False):
        """
        Perform temporal NMS

        !! Check manually if done right after doing it and saving as pkl
        """

        if no_reg:
            all_props = self.props
        else:
            all_props = self._perform_regression()

        all_new_props = []; all_new_scores = []
        K = self.scores[0].shape[1] # num of steps
        for vid_idx, props in enumerate(all_props):
            keep = set()
            N, K = self.scores[vid_idx].shape

            for k in range(K):
                detections = np.zeros((N, 3))
                detections[:, 0] = props[:, 0]
                detections[:, 1] = props[:, 1]
                detections[:, 2] = self.scores[vid_idx][:, k]
                keep |= set(self._perform_temporal_nms(detections, thres))

            keep = list(keep)
            # print (keep, len(keep), len(props))
            # exit(0)
            new_props = props[keep, :]
            new_scores = self.scores[vid_idx][keep, :]
            all_new_props.append(new_props)
            all_new_scores.append(new_scores)

        return all_new_props, all_new_scores


class TaskPoolHead(nn.Module):
    """
    Task head operating over pooled step scores
    """
    def __init__(self, num_steps, num_tasks, middle_layers=[], init_std=0.001):
        super(TaskPoolHead, self).__init__()

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
        for fc in self.fcs:
            scores = fc(scores)
        return scores

class Trainer():
    """
    For training a Network for predicting task
    """

    def __init__(self, model_cfg):
        """
        model_cfg: Configuration for the model 
        """

        self.net_type = model_cfg['type']
        self.num_steps = model_cfg['num_steps']
        self.num_tasks = model_cfg['num_tasks']

        assert self.net_type in ['mlp']
        if self.net_type == 'mlp':
            self.net = TaskPoolHead(self.num_steps, self.num_tasks, model_cfg['middle_layers'])
            self.pooling = model_cfg['pooling']
            assert self.pooling in ['mean', 'max']

        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.net = self.net.cuda()

    def train(self, train_cfg, work_dir, train_data, val_data=None):
        """
        Train the network
        """
        log_file = os.path.join(work_dir, datetime.now().strftime("%d-%m-%Y_%I-%M-%S_%p") + '.log')
        with open(log_file, 'w') as f:
            f.write('')
        for pth_file in glob.glob(os.path.join(work_dir, '*.pth')):
            os.remove(pth_file)

        if self.net_type == 'mlp':
            def gen_dataset(scores, task_ids):
                data = []
                for s in scores:
                    if self.pooling == 'max':
                        datum = np.max(s, axis=0)
                    else:
                        datum = np.mean(s, axis=0)
                    data.append(datum)

                data = torch.Tensor(data)
                task_ids = torch.Tensor(task_ids).type(torch.LongTensor)
                dataset = TensorDataset(data, task_ids)

                return dataset

            train_dataset = gen_dataset(train_data['scores'], train_data['task_ids'])    
            trainloader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)

            if val_data is not None:
                val_dataset = gen_dataset(val_data['scores'], val_data['task_ids'])
                valloader = DataLoader(val_dataset)

            self.net.train()
            criterion = nn.CrossEntropyLoss()

            val_loss = 'NA'
            lr = train_cfg['lr']
            for epoch in range(train_cfg['epochs']):
                if (epoch + 1) % train_cfg['decay'] == 0:
                    lr /= 3
                optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
                
                tot_loss = 0.0
                for data in trainloader:
                    step_scores, task_ids = data
                    if self.cuda_flag:
                        step_scores = step_scores.cuda()
                        task_ids = task_ids.cuda()

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward + backward + optimize
                    out = self.net(step_scores)
                    loss = criterion(out, task_ids)
                    tot_loss += loss.item()

                    loss.backward()
                    optimizer.step()
                
                tot_loss /= len(trainloader)

                if val_data is not None and (epoch + 1) % train_cfg['freq'] == 0:
                    # Update validation loss
                    val_loss = 0.0
                    for data in valloader:
                        step_scores, task_ids = data
                        if self.cuda_flag:
                            step_scores = step_scores.cuda()
                            task_ids = task_ids.cuda()

                        out = self.net(step_scores)
                        loss = criterion(out, task_ids)
                        val_loss += loss.item()

                    val_loss /= len(valloader)

                # logging statistics
                timestamp = str(datetime.now()).split('.')[0]
                val_loss_str = val_loss if type(val_loss) is not float else "{:.3f}".format(val_loss)
                log = '{} | Epoch: {: 3}, lr: {:.4f}, Train Loss: {:.3f}, Val Loss: {}'.format(timestamp, epoch+1, lr, tot_loss, val_loss_str)
                with open(log_file, 'a') as f:
                    f.write('{}\n'.format(log))
                if train_cfg['logging']:
                    print (log)
                if (epoch + 1) % train_cfg['freq'] == 0:
                    # Save the model checkpoint
                    model_file = os.path.join(work_dir, 'epoch_{}.pth'.format(epoch+1))
                    self.save_model(model_file)

    def predict (self, scores, props=None):
        """
        Predict the task score
        """            
        if self.net_type == 'mlp':
            data = []
            for s in scores:
                if self.pooling == 'max':
                    datum = np.max(s, axis=0)
                else:
                    datum = np.mean(s, axis=0)
                data.append(datum)

            inputs = torch.FloatTensor(data)
            if self.cuda_flag:
                inputs = inputs.cuda()

            self.net.eval()
            with torch.no_grad():
                task_scores = self.net(inputs)
            
        return np.argmax(task_scores.cpu().numpy(), axis=1)

    def save_model(self, checkpoint_path):
        torch.save(self.net.state_dict(), checkpoint_path)
    
    def load_model(self, checkpoint_path):
        if self.cuda_flag:
            self.net.load_state_dict(torch.load(checkpoint_path))
        else:
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))

        # Let's print the weights
        # for fc in self.net.fcs:
            # print('..............')
            # print (fc.weight)
            # print (fc.bias)

