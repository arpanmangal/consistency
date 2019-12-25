# Getting Started

This document provides basic tutorials for the usage of MMAction.
For installation, please refer to [INSTALL.md](https://github.com/arpanmangal/consistency/blob/master/INSTALL.md).
For data deployment, please refer to [DATASET.md](https://github.com/arpanmangal/consistency/blob/master/DATASET.md).


## An example on COIN
We first give an example of testing and training temporal action detection models on COIN.
### 1. Prepare data
First of all, please follow [PREPARING_COIN.md](https://github.com/arpanmangal/consistency/blob/master/data_tools/coin/PREPARING_COIN.md) for data preparation.

### 2. Train a model with multiple GPUSs

Use the following training script to train the model:
```
./tools/dist_train_localizer.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
* ${CONFIG_FILE} is the config file stored in `$MMACTION/configs`.
* ${GPU_NUM} is the number of GPU (default: 8). If you are using number other than 8, please adjust the learning rate in the config file linearly.


### 3. Inference with trained models

Testing script for the dataset:
```
python tools/test_localizer.py ${CONFIG_FILE} ${CHECKPOINT_FILE} [--out ${RESULT_FILE}] [other task-specific arguments]
```

-------------------------------------------------------------


## Advanced Usage

Additionally the following scripts could be used to test the model at various checkpoints, and analyze results for different epochs, improve the scores and much more..

### 1. Testing at different epochs
```
python tools/analyze.py test ${MODEL_DIR} ${RESULT_DIR} ${NUM_GPUS} [--start ${START_EPOCH} --step ${STEP_SIZE}]
```
* ${MODEL_DIR} is path of directory contaning the model checkpoints. Expects corresponding `config.py` inside the directory.
* ${RESULT_DIR} is the path of directory where to save the generated result `.pkl` files.
* ${NUM_GPUS} is the number of GPUs used while training.
* ${START_EPOCH} is the epoch from which to start evaluation. (default: 10)
* ${STEP_SIZE}: Evaluate after each ${STEP_SIZE} epochs. (default: 5)

### 2. Pruning the proposals (optional)
Throw away too long / short proposals for better precision.
```
python tools/analyze.py prune ${RESULT_DIR} ${PRUNE_DIR} [--l ${LOW_RANGE} --h ${HI_RANGE}]
```
* ${RESULT_DIR} is the path of the directory containing result `.pkl` files generated above.
* ${PRUNE_DIR} is the path of the directory, where to store the `.pkl` files corresponding to pruned proposals.
* ${LOW_RANGE} is the fraction which if greater than proposal length, leads to discarding the latter. (default: 0.05)
* ${HI_RANGE} is the fraction which if lower than proposal length, lead to discarding the latter. (default: 0.6)

### 3. Enforcing Simple Task Consistency
Enforce TC using domain knowledge in form of a belongingness matrix.
```
python tools/analyze.py tc ${RESULT_DIR} ${RESULT_TC_DIR} [--pooling ${POOLING_TYPE}]
```

* ${RESULT_DIR} is the path of the directory containing result `.pkl` files generated above.
* ${RESULT_TC_DIR} is the path of the directory where to store `.pkl` files corresponding to TC scores.
* ${POOLING_TYPE} is the type of pooling to use. Should be from [`mean` or `max`]. (default: `mean`)

### 4. Evaluate
Evaluate the result pickle files.
```
python tools/analyze.py eval ${MODEL_DIR} ${RESULT_DIR} ${EVAL_DIR}
```
* ${MODEL_DIR} is path of directory contaning the model checkpoints. Expects corresponding `config.py` inside the directory.
* ${RESULT_DIR} is the path of the directory containing result `.pkl` files generated above.
* ${EVAL_DIR} is the directory where to save evaluation results corresponding to each `.pkl` file.

### 5. Parse Results
Parse the results of the evaluation above.
```
python tools/analyze.py parse ${EVAL_DIR}
```
* ${EVAL_DIR} is the directory where the evaluation results generated above are saved. The generated score dict is saved in this directory as well.

### 6. Visualization
Plotting various scores for comparision.
```
python tools/analyze.py plot ${PLOT_TYPE} [--${EVAL_DIRS} --${LABELS} --${SAVE_PATH} --${TITLE}]
```
* ${PLOT_TYPE}: 'key' in the score JSON, which to plot.
* ${EVAL_DIRS}: list of eval directories generated above, each having a `scores.json`.
* ${LABELS}: list of labels for each line curve.
* ${SAVE_PATH}: path to save the plot.
* ${TITLE}: title of the plot