"""
Predicts the action (task) class using the TAG files
Simply takes the max class corresponding to the step IDs of the TAG block
"""
import json

train_tag_file = "~/consistency/data/coin/coin_tag_train_proposal_list.txt"
test_tag_file = "~/consistency/data/coin/coin_tag_test_proposal_list.txt"
coin_json_file = "~/consistency/data/coin/COIN.json"

with open(coin_json_file) as json_file:
    COIN = json.load(json_file)['database']

video_task_map = dict() # Mapping from video ID to task ID
step_task_map = dict() # Mapping from step ID to task ID

for vid_id, vid in COIN.items():
    task_id = vid['recipe_type']
    video_task_map[vid_id] = task_id
    for ann in vid['annotation']:
        step_id = ann['id']
        if step_id not in step_task_map:
            step_task_map[step_id] = task_id
        else:
            print (task_id, step_id)
            print (step_task_map[step_id])
            raise ValueError("Inconsistent Task IDs!!")

print (len(video_task_map))
print (len(step_task_map))