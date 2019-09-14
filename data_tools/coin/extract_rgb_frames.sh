#!/bin/bash

cd ..
python build_rawframes.py ../data/coin/videos ../data/coin/raw_rgb_frames --level 2 --df_path /home/arpan/BTP/coinaction/third_party/dense_flow --ext mp4
echo "Raw frames (RGB only) generated"

cd coin
