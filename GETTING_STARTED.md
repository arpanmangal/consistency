# Getting Started

This document provides basic tutorials for the usage of MMAction.
For installation, please refer to [INSTALL.md](https://github.com/arpanmangal/consistency/blob/master/INSTALL.md).
For data deployment, please refer to [DATASET.md](https://github.com/arpanmangal/consistency/blob/master/DATASET.md).


## An example on COIN
We first give an example of testing and training temporal action detection models on COIN.
### 1. Prepare data
First of all, please follow [PREPARING_COIN.md](https://github.com/arpanmangal/consistency/blob/master/data_tools/coin/PREPARING_COIN.md) for data preparation.

### 2. Generate sliding window proposals

```
time python gen_sliding_window_proposals.py training rgb data/coin/subset_frames data/coin/coin_sw_train_proposal_list.txt --dataset coin
time python gen_sliding_window_proposals.py testing rgb data/coin/subset_frames data/coin/coin_sw_test_proposal_list.txt --dataset coin
```
*Time Needed*: _Negligible_


### 3. Training binary actionness classifier
```
time python binary_train.py coin RGB -b 16 --lr_steps 3 6 --epochs 7 --gpus 0
```
*Time Needed*: _64 mins for 1 domain (Plants and Fruits) (7 tasks)_ 

### 4. Obtaining actionness score
```
time python binary_test.py coin RGB training _rgb_model_best.pth.tar data/coin/rgb_actioness.pkl
time python binary_test.py coin RGB testing _rgb_model_best.pth.tar data/coin/rgb_actioness_test.pkl
```

*Time Needed*: _Plants and Fruits -- Train 67 mins_  
*Time Needed*: _Plants and Fruits -- Test 39 mins_


### 5. Generating TAG Proposals
```
time python gen_bottom_up_proposals.py data/coin/rgb_actioness.pkl --dataset coin --subset training --write_proposals data/coin/coin_tag_train_proposal_list.txt  --frame_path data/coin/subset_frames/
time python gen_bottom_up_proposals.py data/coin/rgb_actioness_test.pkl --dataset coin --subset testing --write_proposals data/coin/coin_tag_test_proposal_list.txt  --frame_path data/coin/subset_frames/
```

*Time Needed*: _Negligible_


### 6. Training SSN
#### 6.1 With ImageNet pretrained
```
time python ssn_train.py coin RGB -b 4 --lr_steps 3 6 --epochs 7 --gpus 0
```
*Time Needed*: _Plants and Fruits -- Train 178 mins_  


### 7. Evaluating on the dataset
#### 7.1 Extract detection scores for all proposals
```
time python ssn_test.py coin RGB ssn_coin_BNInception_rgb_checkpoint.pth.tar data/coin/rgb_plants.pkl --gpus 0
time python ssn_test.py coin RGB _rgb_model_best.pth.tar data/coin/rgb_plants_best.pkl --gpus 0
```
*Time Taken*: _end: 49 mins, best: 49 mins_

#### 7.2 Evaluate detection performance
```
time python eval_detection_results.py coin data/coin/rgb_plants.pkl
time python eval_detection_results.py coin data/coin/rgb_plants_best.pkl
```

**MaP Scores**
Method | 0.1 | 0.2 | 0.3 | 0.4 | 0.5
--- | --- | --- | --- | --- | ---
*SSN* | 1 | 2 | 1 | 2 | 3