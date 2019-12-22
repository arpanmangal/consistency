"""
Networks for task head
Predict task_id from the step scores
"""

import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

    def __init__(self, num_steps, num_tasks, net_type, config):
        """
        net_type: type of network MLP / CNN / RNN
        config: dict() containing the reqd configs
        """

        assert net_type in ['mlp']
        self.net_type = net_type
        if net_type == 'mlp':
            self.net = TaskPoolHead(num_steps, num_tasks, config['middle_layers'])
            self.net.init_weights()
            self.lr = config['lr']
            self.pooling = config['pooling']
            self.epochs = config['epochs']
            self.decay = config['decay']
            self.batch_size = config['batch_size']
            self.log_file = config['log_file']

            assert self.pooling in ['mean', 'max']

        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.net = self.net.cuda()

    def train(self, scores, task_ids, props=None, val_data=None):
        """
        Train the network
        """

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

            train_dataset = gen_dataset(scores, task_ids)    
            trainloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            if val_data is not None:
                val_dataset = gen_dataset(val_data['scores'], val_data['task_ids'])
                valloader = DataLoader(val_dataset)

            self.net.train()
            criterion = nn.CrossEntropyLoss()

            for epoch in range(self.epochs):
                if (epoch + 1) % self.decay == 0:
                    self.lr /= 3
                optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
                
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

                val_loss = 0.0
                if val_data is not None:
                    for data in valloader:
                        step_scores, task_ids = data
                        if self.cuda_flag:
                            step_scores = step_scores.cuda()
                            task_ids = task_ids.cuda()

                        out = self.net(step_scores)
                        loss = criterion(out, task_ids)
                        val_loss += loss.item()

                # logging statistics
                timestamp = str(datetime.datetime.now()).split('.')[0]
                log = '{} | Epoch: {}, lr: {}, Train Loss: {}, Val Loss: {}'.format(timestamp, epoch+1, self.lr, tot_loss, val_loss)
                print (log)
                if self.log_file is not None:
                    with open(self.log_file, 'a') as f:
                        f.write('{}\n'.format(log))

    def predict (self, scores, props=None):
        """
        Predict the task score
        """            
        if self.net_type == 'mlp':
            data = []
            for s in scores:
                if self.pooling == 'max':
                    datum = torch.max(s, dim=0)
                else:
                    datum = torch.mean(s, dim=0)
                data.append(datum)

            inputs = torch.FloatTensor(data)
            if self.cuda_flag:
                inputs = inputs.cuda()

            self.net.eval()
            with torch.no_grad():
                task_scores = self.net(inputs)
            
        return np.argmax(task_scores, axis=1)

    def save_model(self, checkpoint_path):
        info = {
            'model': self.net.state_dict(),
            'pooling': self.pooling,
            'type': self.net_type
        }
        torch.save(info, checkpoint_path)
    
    def load_model(self, checkpoint_path):
        if self.cuda_flag:
            info = torch.load(checkpoint_path)
        else:
            info = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        self.net.load_state_dict(info['model'])
        assert self.pooling == info['pooling'].cpu()
        assert self.net_type == info['type'].cpu()

