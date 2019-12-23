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

    def train(self, train_cfg, scores, task_ids, props=None, val_data=None):
        """
        Train the network
        """

        with open(train_cfg['log_file'], 'w') as f:
            f.write('')

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
            trainloader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'], shuffle=True)

            if val_data is not None:
                val_dataset = gen_dataset(val_data['scores'], val_data['task_ids'])
                valloader = DataLoader(val_dataset)

            self.net.train()
            criterion = nn.CrossEntropyLoss()

            val_loss = float('inf')
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
                timestamp = str(datetime.datetime.now()).split('.')[0]
                log = '{} | Epoch: {}, lr: {:.4f}, Train Loss: {:.3f}, Val Loss: {:.3f}'.format(timestamp, epoch+1, lr, tot_loss, val_loss)
                print (log)
                if train_cfg['log_file'] is not None:
                    with open(train_cfg['log_file'], 'a') as f:
                        f.write('{}\n'.format(log))

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
        for fc in self.net.fcs:
            print('..............')
            print (fc.weight)
            print (fc.bias)

