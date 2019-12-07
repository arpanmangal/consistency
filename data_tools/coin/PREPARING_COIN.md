## Preparing COIN

For more details, please refer to the [official website](https://coin-dataset.github.io/). We provide scripts with documentations. Before we start, please make sure that the directory is located at `$CONSISTENCY/data_tools/coin/`.

## Download from the COIN repo: 
### Prepare annotations
Run the following script to download the annotations.
```
python download_annotations.py
```

### Prepare videos
Run the following script to prepare videos. 
```
python download_videos.py
```

### Extract frames
Now it is time to extract frames from videos. 
Before extraction, please refer to `DATASET.md` for installing [dense_flow](https://github.com/yjxiong/dense_flow).
If you have some SSD, then we strongly recommend extracting frames there for better I/O performance. 

This is expected to take some time.
```shell
# execute these two line (Assume the SSD is mounted at "/mnt/SSD/")
mkdir /mnt/SSD/coin_extracted/
ln -s /mnt/SSD/coin_extracted/ ../data/coin/raw_frames/
```
Afterwards, run the following script to extract frames.
```shell
bash extract_frames.sh
```

-------------------------------------------------------------
### Split the data into training, validation and testing

Given dataset only has train+test split, so run the following script to split the data.

```
python prepare_split.py --root <PATH_OF_THIS_REPO>
```

After doing hyperparameter tuning, train on full train-set. Merge the validation split into the train split.
```
python prepare_split.py --root <PATH_OF_THIS_REPO> --no_val 
```

-------------------------------------------------------

### Prepare subset of data for training
For initial experiments, it is recommended to work with a small subset of the dataset. We work with a subset consisting of selected tasks out of the total 180 tasks.

Firstly, put down the `task` ids (folder numbers from the downloaded videos folder) in the file `subset`, with each id in a new line. Then run the following to generate a subset of the dataset.

```
python prepare_coin.py
```

This code will generate the `json` file for the subset of the dataset, mapping the original `task ID`s to new IDs starting from 0. It will also create soft links to respective videos and extracted frames in the folders `subset` and `subset_frames` respectively.

----------------------------------

<!-- ### Fetching proposal files
Run the follow scripts to fetch pre-computed tag proposals.
```shell
bash fetch_tag_proposals.sh
``` -->

### Folder structure
In the context of the whole project (for coin only), the folder structure will look like: 

```
consistency
├ data
│   ├ coin
│   │   │ coin_tag_val_normalized_proposal_list.txt
│   │   │ coin_tag_test_normalized_proposal_list.txt
│   │   │ COIN_full.json
│   │   │ COIN.json
│   │   │ videos
│   │   │   │ 30
│   │   │   │   ├ 0JHFaCl9Zb4.mp4
│   │   │   │   ├ 0OsP1icacOY.mp4
│   │   │   │   ├ ...
│   │   │   │ 114
│   │   │   │   ├ 0lY2qrtPZR0.mp4
│   │   │   │   ├ ...
│   │   │   │ ...
│   │   │ subset
│   │   │   ├ 0
│   │   │   │   ├ 0JHFaCl9Zb4.mp4
│   │   │   │   ├ 0OsP1icacOY.mp4
│   │   │   │   ├ ...
│   │   │   │ 1
│   │   │   │   ├ 0lY2qrtPZR0.mp4
│   │   │   │   ├ ...
│   │   │   │ ...
│   │   │ raw_frames
│   │   │   │ 30
│   │   │   │   │ 0JHFaCl9Zb4
│   │   │   │   │   ├ img_00001.jpg
│   │   │   │   │   ├ img_00002.jpg
│   │   │   │   │   ├ ...
│   │   │   │   │   ├ flow_x_00001.jpg
│   │   │   │   │   ├ flow_x_00002.jpg
│   │   │   │   │   ├ ...
│   │   │   │   │   ├ flow_y_00001.jpg
│   │   │   │   │   ├ flow_y_00002.jpg
│   │   │   │   │   ├ ...
│   │   │   │   │ ...
│   │   │   │ 14
│   │   │   │   │ 0lY2qrtPZR0
│   │   │   │   │   ├ img_00001.jpg
│   │   │   │   │   ├ img_00002.jpg
│   │   │   │   │   ├ ...
│   │   │   │   │   ├ flow_x_00001.jpg
│   │   │   │   │   ├ flow_x_00002.jpg
│   │   │   │   │   ├ ...
│   │   │   │   │   ├ flow_y_00001.jpg
│   │   │   │   │   ├ flow_y_00002.jpg
│   │   │   │   │   ├ ...
│   │   │   │   │ ...
│   │   │   │ ...
│   │   │ subset_frames
│   │   │   │ 0JHFaCl9Zb4
│   │   │   │   ├ img_00001.jpg
│   │   │   │   ├ img_00002.jpg
│   │   │   │   ├ ...
│   │   │   │   ├ flow_x_00001.jpg
│   │   │   │   ├ flow_x_00002.jpg
│   │   │   │   ├ ...
│   │   │   │   ├ flow_y_00001.jpg
│   │   │   │   ├ flow_y_00002.jpg
│   │   │   │   ├ ...
│   │   │   │ 0lY2qrtPZR0
│   │   │   │   ├ img_00001.jpg
│   │   │   │   ├ img_00002.jpg
│   │   │   │   ├ ...
│   │   │   │   ├ flow_x_00001.jpg
│   │   │   │   ├ flow_x_00002.jpg
│   │   │   │   ├ ...
│   │   │   │   ├ flow_y_00001.jpg
│   │   │   │   ├ flow_y_00002.jpg
│   │   │   │   ├ ...
│   │   │   │ ...
```

For training and evaluating on COIN, please refer to [GETTING_STARTED.md](https://github.com/arpanmangal/consistency/blob/master/GETTING_STARTED.md).
