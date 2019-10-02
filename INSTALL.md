# Installation

```shell
git clone --recursive https://github.com/arpanmangal/consistency
```

## Requirements

- Linux
- Python 3.5+
- PyTorch <= 0.3
- Torchvision 0.1.8
- CUDA 9.0+
- NVCC 2+
- GCC 7.4+
- [dense_flow](https://github.com/yjxiong/dense_flow)
- [mmcv](https://github.com/open-mmlab/mmcv).

Note that "dense_flow" will be contained in this codebase as a submodule. (TODO)

## Install Pre-requisites

### Basic requirements
```
pip install requirements.txt
```

### Install dense_flow
[Dense_flow](https://github.com/yjxiong/dense_flow) is used to calculate the optical flow of videos.
If you just want to have a quick experience with MMAction without taking pain of installing opencv, you can skip this step.

Note that Dense_flow now supports OpenCV 4.1.0, 3.1.0 and 2.4.13.
The master branch is for 4.1.0. For those with 2.4.13, please refer to the lines with strikethrough.

<del>
(a) Install OpenCV=2.4.13

```shell
cd third_party/
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils

wget -O OpenCV-2.4.13.zip https://github.com/Itseez/opencv/archive/2.4.13.zip
unzip OpenCV-2.4.13.zip

cd opencv-2.4.13
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON -D WITH_V4L=ON ..
make -j32
cd ../../../
```
</del>

(a) Install OpenCV=4.1.0

0. (For CUDA 10.0 only) CUDA 9.x should have no problem.
  Video decoder is deprecated in CUDA 10.0.
To handle this, download [NVIDIA VIDEO CODEC SDK](https://developer.nvidia.com/nvidia-video-codec-sdk) and copy the header files to your cuda path (`/usr/local/cuda-10.0/include/` for example).
Note that you may have to do as root.

```shell
unzip Video_Codec_SDK_9.0.20.zip
cp Video_Codec_SDK_9.0.20/include/nvcuvid.h /usr/local/cuda-10.0/include/
cp Video_Codec_SDK_9.0.20/include/cuviddec.h /usr/local/cuda-10.0/include/
cp Video_Codec_SDK_9.0.20/Lib/linux/stubs/x86_64/libnvcuvid.so /usr/local/cuda-10.0/lib64/libnvcuvid.so.1
```

1. Obtain required packages for building OpenCV 4.1.0 (duplicated with requirements for Decord in part)

```shell
sudo apt-get install -y liblapack-dev libatlas-base-dev
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev
sudo apt-get install -y software-properties-common
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt update
sudo apt install -y ffmpeg
```

2. Obtain OpenCV 4.1.0 and its extra modules (optflow, etc.) by

```shell
cd third_party
wget -O OpenCV-4.1.0.zip wget https://github.com/opencv/opencv/archive/4.1.0.zip
unzip OpenCV-4.1.0.zip
wget -O OpenCV_contrib-4.1.0.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip
unzip OpenCV_contrib-4.1.0.zip
```

3. Build OpenCV 4.1.0 from scratch (due to some custom settings)

```
cd opencv-4.1.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules/ -DWITH_TBB=ON -DBUILD_opencv_cnn_3dobj=OFF -DBUILD_opencv_dnn=OFF -DBUILD_opencv_dnn_modern=OFF -DBUILD_opencv_dnns_easily_fooled=OFF -DOPENCV_ENABLE_NONFREE=ON ..
make -j
```

Note that `-DOPENCV_ENABLE_NONFREE=ON` is explicitly set to enable *warped flow* proposed in [TSN](https://arxiv.org/abs/1608.00859).
You can skip this argument to speed up the compilation if you do not intend to use it.

(b) Build dense_flow
```shell
cd third_party/dense_flow
# dense_flow dependencies
sudo apt-get -qq install libzip-dev
mkdir build && cd build
# deprecated:
# OpenCV_DIR=../../opencv-2.4.13/build cmake ..
OpenCV_DIR=../../opencv-4.1.0/build cmake ..
make -j
```

Please refer to [DATASET.md](https://github.com/open-mmlab/mmaction/blob/master/DATASET.md) to get familar with the data preparation and to [GETTING_STARTED.md](https://github.com/open-mmlab/mmaction/blob/master/GETTING_STARTED.md) to use CONSistency.

