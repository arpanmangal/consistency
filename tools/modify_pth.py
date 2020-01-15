"""
Code for combining pytorch model
"""

import torch
import os

work_dir = '/home/cse/btech/cs1160321/consistency/work_dirs'
template_path = os.path.join(work_dir, 'mtl_dmlp/epoch_120.pth')
step_checkpoint = os.path.join(work_dir, '200_400_450/epoch_255.pth')
task_checkpoint = os.path.join(work_dir, 'tn/mlp_double/epoch_320.pth')
combined_checkpoint = os.path.join(work_dir, 'mtl_bestssn/epoch_120.pth')

if torch.cuda.is_available():
    model = torch.load(template_path)
    step_weights = torch.load(step_checkpoint)['state_dict']
    task_weights = torch.load(task_checkpoint)
else:
    model = torch.load(template_path, map_location=torch.device('cpu'))
    step_weights = torch.load(step_checkpoint, map_location=torch.device('cpu'))['state_dict']
    task_weights = torch.load(task_checkpoint, map_location=torch.device('cpu'))

del model['optimizer']
del model['meta']['config']

# transfer the weights
for k, w in step_weights.items():
    model['state_dict'][k] = w

for k, w in task_weights.items():
    model['state_dict']['task_head.' + k] = w

torch.save(model, combined_checkpoint)
