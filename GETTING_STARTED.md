# Getting Started

This document provides basic tutorials for the usage of MMAction.
For installation, please refer to [INSTALL.md](https://github.com/arpanmangal/consistency/blob/master/INSTALL.md).
For data deployment, please refer to [DATASET.md](https://github.com/arpanmangal/consistency/blob/master/DATASET.md).


## An example on COIN
We first give an example of testing and training temporal action detection models on COIN.
### 1. Prepare data
First of all, please follow [PREPARING_COIN.md](https://github.com/arpanmangal/consistency/blob/master/data_tools/ucf101/PREPARING_UCF101.md) for data preparation.

### 2. Generate sliding window proposals

```
python gen_sliding_window_proposals.py training rgb data/coin/subset_frames data/coin/coin_sw_train_proposal_list.txt --dataset coin
python gen_sliding_window_proposals.py testing rgb data/coin/subset_frames data/coin/coin_sw_test_proposal_list.txt --dataset coin

```
