# ReSpeaker ROS2 Audio Recorder

这个仓库用于采集 ReSpeaker 多通道音频，发布 ROS2 topic，录制 ROS2 bag，并把 bag 中的音频导出为 WAV。

主要包：

```text
src/respeaker_ros2_recorder
```

默认 topic：

```text
/respeaker/audio_raw
```

## 1. 环境准备

如果使用 ROS2 Foxy + conda `open-mmlab` 环境，建议每次新开终端后按这个顺序：

```bash
cd /home/kemove/yyz/audio-nav/ws_col
conda activate open-mmlab

unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_PACKAGE_PATH
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH PYTHONPATH

source /opt/ros/foxy/setup.bash
```

检查不要混入 ROS1：

```bash
echo $PYTHONPATH
```

不应该出现：

```text
/opt/ros/noetic
```

安装依赖：

```bash
sudo apt update
sudo apt install ros-foxy-rosbag2 ros-foxy-ros2bag ros-foxy-rosbag2-storage-default-plugins
python -m pip install numpy soundfile pyaudio empy==3.3.4
```

## 2. 编译

用 conda 环境里的 Python 编译：

```bash
colcon build --packages-select respeaker_ros2_recorder --symlink-install \
  --cmake-args \
  -DPYTHON_EXECUTABLE=/home/kemove/anaconda3/envs/open-mmlab/bin/python \
  -DPython3_EXECUTABLE=/home/kemove/anaconda3/envs/open-mmlab/bin/python

source install/setup.bash
```

检查包是否存在：

```bash
ros2 pkg list | grep respeaker_ros2_recorder
```

## 3. 查看音频设备

```bash
arecord -l
ros2 run respeaker_ros2_recorder list_audio_devices.py
```

示例：

```text
Index 2: ReSpeaker 6 Mic Array | maxInputChannels=6 | defaultSampleRate=16000.0
```

`device_index:=-1` 表示使用系统默认输入设备。

## 4. 启动采集

只发布音频 topic：

```bash
ros2 launch respeaker_ros2_recorder respeaker_collect.launch.py device_index:=2
```

检查 topic：

```bash
ros2 topic list
ros2 topic info /respeaker/audio_raw
ros2 topic echo /respeaker/audio_raw --once
```

## 5. 采集并录制 Bag

```bash
ros2 launch respeaker_ros2_recorder respeaker_collect.launch.py \
  device_index:=2 \
  record_bag:=true \
  bag_path:=respeaker_audio
```

停止录制：

```text
Ctrl+C
```

查看 bag：

```bash
ros2 bag info respeaker_audio
```

## 6. 单个 Bag 导出 WAV

导出全部通道：

```bash
ros2 run respeaker_ros2_recorder export_audio_from_bag.py \
  --bag respeaker_audio \
  --topic /respeaker/audio_raw \
  --out respeaker_audio.wav
```

导出指定通道，例如通道 0 和 1：

```bash
ros2 run respeaker_ros2_recorder export_audio_from_bag.py \
  --bag respeaker_audio \
  --topic /respeaker/audio_raw \
  --out respeaker_ch0_ch1.wav \
  --channels 0,1
```

也可以直接指定 `.db3`：

```bash
ros2 run respeaker_ros2_recorder export_audio_from_bag.py \
  --bag respeaker_audio/respeaker_audio_0.db3 \
  --topic /respeaker/audio_raw \
  --out respeaker_audio.wav
```

## 7. 批量导出 WAV

假设有多个 bag：

```text
bags/
├── bag_001/
│   └── bag_001_0.db3
├── bag_002/
│   └── bag_002_0.db3
└── bag_003/
    └── bag_003_0.db3
```

批量导出全部通道：

```bash
ros2 run respeaker_ros2_recorder batch_export_audio_from_bags.py \
  --input-dir bags \
  --output-dir wav_exports \
  --topic /respeaker/audio_raw
```

批量导出指定通道：

```bash
ros2 run respeaker_ros2_recorder batch_export_audio_from_bags.py \
  --input-dir bags \
  --output-dir wav_exports \
  --topic /respeaker/audio_raw \
  --channels 0,1
```

递归搜索多层目录：

```bash
ros2 run respeaker_ros2_recorder batch_export_audio_from_bags.py \
  --input-dir bags \
  --output-dir wav_exports \
  --recursive
```

## 8. 常用参数

```text
sample_rate  默认 16000
channels     默认 6
chunk_size   默认 1600，约 0.1 秒音频
device_index PyAudio 输入设备编号，-1 表示默认设备
topic_name   默认 /respeaker/audio_raw
record_bag   true/false，是否同时录 bag
bag_path     bag 输出目录
```

## 9. 常见问题

包找不到：

```bash
source /opt/ros/foxy/setup.bash
source /home/kemove/yyz/audio-nav/ws_col/install/setup.bash
ros2 pkg list | grep respeaker_ros2_recorder
```

混入 ROS1：

```bash
echo $PYTHONPATH
```

如果看到 `/opt/ros/noetic`，新开终端，重新按第 1 节配置。

`ModuleNotFoundError: rosbag2_py`：

当前导出脚本支持直接读取 `.db3`，不强依赖 `rosbag2_py`。重新编译并 source：

```bash
colcon build --packages-select respeaker_ros2_recorder --symlink-install
source install/setup.bash
```

设备打不开：

```bash
arecord -l
ros2 run respeaker_ros2_recorder list_audio_devices.py
```

确认 `device_index`、`channels`、`sample_rate` 和实际设备一致。

## 10. 最短完整流程

```bash
cd /home/kemove/yyz/audio-nav/ws_col
conda activate open-mmlab
source /opt/ros/foxy/setup.bash

colcon build --packages-select respeaker_ros2_recorder --symlink-install
source install/setup.bash

ros2 run respeaker_ros2_recorder list_audio_devices.py

ros2 launch respeaker_ros2_recorder respeaker_collect.launch.py \
  device_index:=2 \
  record_bag:=true \
  bag_path:=respeaker_audio

ros2 run respeaker_ros2_recorder export_audio_from_bag.py \
  --bag respeaker_audio \
  --topic /respeaker/audio_raw \
  --out respeaker_audio.wav
```
