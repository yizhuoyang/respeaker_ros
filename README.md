# ReSpeaker ROS1 Audio Recorder

这个包用于在 ROS1 中采集 ReSpeaker 多通道音频，发布音频 topic，录制 rosbag，并把 bag 中的音频导出为 WAV。

包名：

```text
respeaker_ros_recorder
```

默认 topic：

```text
/respeaker/audio_raw
```

消息类型：

```text
respeaker_ros_recorder/AudioDataStamped
```

## 1. 环境准备

建议新开一个终端，只 source ROS1 Noetic，不要和 ROS2 Foxy/Humble 混用：

```bash
cd /home/kemove/yyz/audio-nav/ws_col

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_PACKAGE_PATH
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH

source /opt/ros/noetic/setup.bash
```

安装依赖：

```bash
sudo apt update
sudo apt install python3-pyaudio python3-numpy python3-soundfile ros-noetic-rosbag
```

如果你使用 conda，建议 ROS1 采集时先退出 conda：

```bash
conda deactivate
```

## 2. 编译

从工作区根目录编译：

```bash
cd /home/kemove/yyz/audio-nav/ws_col
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

这里显式指定 `PYTHON_EXECUTABLE`，避免 `catkin_make` 误用 conda 或其他 ROS 环境里的 Python。

检查包是否能找到：

```bash
rospack find respeaker_ros_recorder
```

## 3. 查看音频设备

先确认 Linux 能看到声卡：

```bash
arecord -l
```

再查看 PyAudio 输入设备编号：

```bash
rosrun respeaker_ros_recorder list_audio_devices.py
```

示例：

```text
Index 2: ReSpeaker 6 Mic Array | maxInputChannels=6 | defaultSampleRate=16000.0
```

后续启动时使用：

```text
device_index:=2
```

`device_index:=-1` 表示使用系统默认输入设备。

## 4. 启动采集

只发布音频 topic，不录 bag：

```bash
roslaunch respeaker_ros_recorder respeaker_collect.launch device_index:=2
```

检查 topic：

```bash
rostopic list
rostopic info /respeaker/audio_raw
rostopic echo /respeaker/audio_raw -n 1
```

## 5. 采集并录制 Bag

启动采集并同时录制 rosbag：

```bash
roslaunch respeaker_ros_recorder respeaker_collect.launch \
  device_index:=2 \
  record_bag:=true \
  bag_path:=respeaker_audio
```

这会生成：

```text
respeaker_audio.bag
```

停止录制：

```text
Ctrl+C
```

查看 bag：

```bash
rosbag info respeaker_audio.bag
```

## 6. 单个 Bag 导出 WAV

导出全部通道：

```bash
rosrun respeaker_ros_recorder export_audio_from_bag.py \
  --bag respeaker_audio.bag \
  --topic /respeaker/audio_raw \
  --out respeaker_audio.wav
```

只导出指定通道，例如通道 0 和 1：

```bash
rosrun respeaker_ros_recorder export_audio_from_bag.py \
  --bag respeaker_audio.bag \
  --topic /respeaker/audio_raw \
  --out respeaker_ch0_ch1.wav \
  --channels 0,1
```

通道编号从 `0` 开始。6 通道设备的有效编号是：

```text
0,1,2,3,4,5
```

## 7. 双麦克风采集并录制 Bag

`dual_audio_collect.launch` 会同时启动两个音频输入，并把两个 topic 一起录进同一个 bag。默认 topic：

```text
/mic1/audio_raw
/mic2/audio_raw
```

先用下面命令确认两个麦克风的 PyAudio 设备编号：

```bash
rosrun respeaker_ros_recorder list_audio_devices.py
```

启动双麦克风采集并录制：

```bash
roslaunch respeaker_ros_recorder dual_audio_collect.launch \
  mic1_device_index:=2 \
  mic1_channels:=2 \
  mic2_device_index:=3 \
  mic2_channels:=6 \
  record_bag:=true \
  bag_path:=dual_audio
```

这会生成：

```text
dual_audio.bag
```

查看 bag 中是否包含两个音频 topic：

```bash
rosbag info dual_audio.bag
```

## 8. 双麦克风 Bag 导出两个 WAV

从 `dual_audio_collect.launch` 录制的 bag 中一次导出两个音频：

```bash
rosrun respeaker_ros_recorder export_dual_audio_from_bag.py \
  --bag dual_audio.bag \
  --mic1-topic /mic1/audio_raw \
  --mic2-topic /mic2/audio_raw \
  --out-dir dual_wav
```

默认会生成：

```text
dual_wav/dual_audio_mic1.wav
dual_wav/dual_audio_mic2.wav
```

也可以手动指定两个输出文件：

```bash
rosrun respeaker_ros_recorder export_dual_audio_from_bag.py \
  --bag dual_audio.bag \
  --mic1-out mic1.wav \
  --mic2-out mic2.wav
```

如果只想导出指定通道，可以分别给两个麦克风设置通道编号：

```bash
rosrun respeaker_ros_recorder export_dual_audio_from_bag.py \
  --bag dual_audio.bag \
  --mic1-channels 0,1 \
  --mic2-channels 0,1,2,3,4,5 \
  --out-dir dual_wav
```

批量导出一个目录下的 dual bag：

```bash
rosrun respeaker_ros_recorder export_dual_audio_from_bag.py \
  --bag-dir /media/kemove/T9/bag/dual_bag_nav \
  --out-dir /media/kemove/T9/bag/dual_wav_exports \
  --keep-going
```

批量导出默认会为每个 bag 生成：

```text
<bag-name>_mic1.wav
<bag-name>_mic2.wav
```

## 9. 常用参数

```text
sample_rate  默认 16000
channels     默认 6
chunk_size   默认 1600，约 0.1 秒音频
device_index PyAudio 输入设备编号，-1 表示默认设备
frame_id     默认 respeaker
topic_name   默认 /respeaker/audio_raw
record_bag   true/false，是否同时录 bag
bag_path     bag 输出文件名前缀
```

双麦克风参数：

```text
mic1_sample_rate   默认 16000
mic1_channels      默认 2
mic1_chunk_size    默认 1600
mic1_device_index  默认 -1
mic1_frame_id      默认 mic1
mic1_topic_name    默认 /mic1/audio_raw

mic2_sample_rate   默认 16000
mic2_channels      默认 6
mic2_chunk_size    默认 1600
mic2_device_index  默认 -1
mic2_frame_id      默认 mic2
mic2_topic_name    默认 /mic2/audio_raw
```

## 10. 常见问题

包找不到：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/yyz/audio-nav/ws_col/devel/setup.bash
rospack find respeaker_ros_recorder
```

混入 ROS2：

```bash
echo $AMENT_PREFIX_PATH
echo $COLCON_PREFIX_PATH
```

如果看到 Foxy/Humble 的路径，新开终端，重新按第 1 节配置。

设备打不开：

```bash
arecord -l
rosrun respeaker_ros_recorder list_audio_devices.py
```

确认 `device_index`、`channels`、`sample_rate` 和实际设备一致。

导出 WAV 失败：

```bash
python3 -c "import numpy, soundfile, rosbag; print('ok')"
```

如果缺包，安装：

```bash
sudo apt install python3-numpy python3-soundfile
```

## 11. 最短完整流程

```bash
cd /home/kemove/yyz/audio-nav/ws_col
source /opt/ros/noetic/setup.bash
catkin_make -DPYTHON_EXECUTABLE=/home/kemove/anaconda3/envs/open-mmlab/bin/python3
source devel/setup.bash

rosrun respeaker_ros_recorder list_audio_devices.py

roslaunch respeaker_ros_recorder respeaker_collect.launch \
  device_index:=2 \
  record_bag:=true \
  bag_path:=respeaker_audio

rosrun respeaker_ros_recorder export_audio_from_bag.py \
  --bag respeaker_audio.bag \
  --topic /respeaker/audio_raw \
  --out respeaker_audio.wav



rosrun respeaker_ros_recorder batch_export_audio_from_bags.py \
  --bag-dir /media/kemove/T9/bag/bag_nav \
  --topic /respeaker/audio_raw \
  --keep-going
```

双麦克风最短流程：

```bash
rosrun respeaker_ros_recorder list_audio_devices.py

roslaunch respeaker_ros_recorder dual_audio_collect.launch \
  mic1_device_index:=2 \
  mic1_channels:=2 \
  mic2_device_index:=3 \
  mic2_channels:=6 \
  record_bag:=true \
  bag_path:=dual_audio

rosrun respeaker_ros_recorder export_dual_audio_from_bag.py \
  --bag dual_audio.bag \
  --out-dir dual_wav
```

批量导出 dual bag：

```bash
rosrun respeaker_ros_recorder export_dual_audio_from_bag.py \
  --bag-dir /media/kemove/T9/bag/dual_bag_nav \
  --keep-going
```
