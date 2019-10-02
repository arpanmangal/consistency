## Dataset Preparation

### Supported datasets

Currently we only suuport the [COIN](https://coin-dataset.github.io/) dataset. See [PREPARING_COIN.md](https://github.com/arpanmangal/consistency/blob/master/data_tools/coin/PREPARING_COIN.md) for data preparation scripts.

Now, you can switch to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/tree/master/GETTING_STARTED.md) to train and test the model.


**TL;DR** The following guide is helpful when you want to experiment with custom dataset.
Similar to the datasets stated above, it is recommended organizing in `$CONSISTENCY/data/$DATASET`.

### Prepare annotations

### Prepare videos
Please refer to the official website and/or the official script to prepare the videos.
Note that the videos should be arranged in either (1) a two-level directory organized by `${CLASS_NAME}/${VIDEO_ID}` or (2) a single-level directory.
It is recommended using (1) for action recognition datasets (such as UCF101 and Kinetics) and using (2) for action detection datasets or those with multiple annotations per video (such as THUMOS14 and AVA).


### Extract frames
To extract frames (optical flow, to be specific), [dense_flow](https://github.com/yjxiong/dense_flow) is needed.
(**TODO**: This will be merged into MMAction in the next version in a smoother way).
For the time being, please use the following command:

```shell
python build_rawframes.py $SRC_FOLDER $OUT_FOLDER --df_path $PATH_OF_DENSE_FLOW --level {1, 2}
```
- `$SRC_FOLDER` points to the folder of the original video (for example)
- `$OUT_FOLDER` points to the root folder where the extracted frames and optical flow store 
- `$PATH_OF_DENSE_FLOW` points to the root folder where dense_flow is installed.
- `--level` is either 1 for the single-level directory or 2 for the two-level directory

The recommended practice is

1. set `$OUT_FOLDER` to be an folder located in SSD
2. symlink the link `$OUT_FOLDER` to `$MMACTION/data/$DATASET/rawframes`.

```shell
ln -s ${OUT_FOLDER} $MMACTION/data/$DATASET/rawframes
```

### Generate filelist
```shell
cd $MMACTION
python data_tools/build_file_list.py ${DATASET} ${SRC_FOLDER} --level {1, 2} --format {rawframes, videos}
```
- `${SRC_FOLDER}` should point to the folder of the corresponding to the data format:
    - "$MMACTION/data/$DATASET/rawframes" `--format rawframes`
    - "$MMACTION/data/$DATASET/videos" if `--format videos`
