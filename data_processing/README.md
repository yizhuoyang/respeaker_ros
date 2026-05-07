# ROS2 Bag Data Extraction

这个目录用于把 ROS2 bag 中的图像、里程计和 Livox 点云提取到普通文件夹。

脚本：

```text
extract_ros2_bag_data.py
sync_audio_with_extracted_data.py
sync_from_bag_and_audio.py
```

默认提取这些 topic：

```text
/camera/color/image_raw
/camera/depth/image_raw
/lio/odom
/lio/robo/odom
/livox/lidar
```

## 使用前环境

先 source ROS2 和你的工作区。如果 bag 里有 Livox 自定义消息，也要 source 能找到 `livox_ros_driver2` 的工作区。

```bash
cd /home/kemove/yyz/audio-nav/ws_col
conda activate open-mmlab
source /opt/ros/foxy/setup.bash
source install/setup.bash
```

如果希望图像保存为 PNG，建议安装 OpenCV：

```bash
python -m pip install opencv-python
```

没有 OpenCV 时，图像会保存为 `.npy`。

## 提取单个 Bag

输入可以是 bag 目录：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input /path/to/rosbag2_xxx \
  --output extracted_data
```

也可以是 `.db3` 文件：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input /path/to/rosbag2_xxx/rosbag2_xxx_0.db3 \
  --output extracted_data
```

## 批量提取多个 Bag

假设目录结构是：

```text
bags/
├── bag_001/
│   └── bag_001_0.db3
├── bag_002/
│   └── bag_002_0.db3
└── bag_003/
    └── bag_003_0.db3
```

运行：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input bags \
  --output extracted_data
```

如果 bag 在更深层目录：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input bags \
  --output extracted_data \
  --recursive
```

## 只提取部分 Topic

可选标签：

```text
color
depth
lio_odom
lio_robo_odom
livox
```

例如只提取彩色图像、机器人 odom 和 Livox：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input bags \
  --output extracted_data \
  --topics color,lio_robo_odom,livox
```

只提取图像：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input bags \
  --output extracted_data \
  --topics color,depth
```

## 只提取某个时间段

`--start-ns` 和 `--end-ns` 使用 ROS2 bag 数据库中的 `timestamp`，单位是纳秒。

```bash
python data_processing/extract_ros2_bag_data.py \
  --input bags \
  --output extracted_data_clip \
  --topics color,depth,lio_robo_odom \
  --start-ns 1778038000000000000 \
  --end-ns 1778038010000000000
```

## 输出结构

每个 bag 会生成一个同名文件夹：

```text
extracted_data/
└── bag_001/
    ├── topics_manifest.json
    ├── color/
    │   ├── index.csv
    │   └── 000001_<timestamp_ns>.png
    ├── depth/
    │   ├── index.csv
    │   └── 000001_<timestamp_ns>.png 或 .npy
    ├── lio_odom.csv
    ├── lio_robo_odom.csv
    └── livox/
        ├── index.csv
        └── 000001_*.csv
```

说明：

- `color/`：彩色图像。
- `depth/`：深度图像。`32FC1` 深度会保存为 `.npy`。
- `lio_odom.csv`：`/lio/odom` 的位姿和速度。
- `lio_robo_odom.csv`：`/lio/robo/odom` 的位姿和速度。
- `livox/`：每帧 Livox 点云一个 CSV。
- `topics_manifest.json`：bag 中所有 topic 和类型信息。

第一步提取时，图像和 Livox 帧文件名都会包含消息时间戳，例如：

```text
000001_1778038000123456789.png
```

同时 `index.csv` 中也保存了 `bag_timestamp_ns` 和消息 header 时间戳，方便第二步同步。

## 用 WAV 生成同步样本

如果你已经有同一段录制的音频 WAV，可以把音频切成小段，然后为每段匹配最近的图像和 odom。

音频文件名如果是下面这种格式：

```text
clock1_1778111042843891356.wav
```

脚本会自动从文件名里读取 `1778111042843891356` 作为音频开始时间戳，不需要再手动传 `--audio-start-ns`。

先完成第一步提取：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input bags/bag_001 \
  --output extracted_data \
  --topics color,depth,lio_odom,lio_robo_odom
```

再生成同步样本：

```bash
python data_processing/sync_audio_with_extracted_data.py \
  --extracted extracted_data/bag_001 \
  --audio audio/clock1_1778111042843891356.wav \
  --output synced_dataset/bag_001 \
  --segment-sec 1.0 \
  --hop-sec 1.0
```

如果只想处理某个时间段：

```bash
python data_processing/sync_audio_with_extracted_data.py \
  --extracted extracted_data/bag_001 \
  --audio audio/clock1_1778111042843891356.wav \
  --start-ns 1778038005000000000 \
  --end-ns 1778038015000000000 \
  --output synced_dataset/bag_001 \
  --segment-sec 0.5 \
  --hop-sec 0.5
```

输出结构：

```text
synced_dataset/
└── bag_001/
    ├── dataset_manifest.json
    ├── audio/
    │   ├── 000000.wav
    │   └── 000001.wav
    ├── color/
    │   ├── 000000.png
    │   └── 000001.png
    ├── depth/
    │   ├── 000000.png 或 .npy
    │   └── 000001.png 或 .npy
    ├── lio_odom/
    │   ├── 000000.npy
    │   └── 000001.npy
    ├── lio_robo_odom/
    │   ├── 000000.npy
    │   └── 000001.npy
    ├── lio_odom.npz
    ├── lio_robo_odom.npz
    └── metadata/
        ├── 000000.json
        └── 000001.json
```

这里的 `000000`、`000001` 是同步后的样本编号。每种模态单独一个目录，方便后续训练代码按编号读取。

同步后的 odom 保存为 NumPy 格式：

- `lio_odom/000000.npy`：单个样本的 odom 向量。
- `lio_robo_odom/000000.npy`：单个样本的机器人 odom 向量。
- `lio_odom.npz`：所有样本的 `/lio/odom` 汇总。
- `lio_robo_odom.npz`：所有样本的 `/lio/robo/odom` 汇总。

odom 向量字段顺序：

```text
px, py, pz,
qx, qy, qz, qw,
linear_x, linear_y, linear_z,
angular_x, angular_y, angular_z
```

读取 `.npz` 示例：

```python
import numpy as np

odom = np.load("synced_dataset/bag_001/lio_robo_odom.npz")
print(odom["fields"])
print(odom["data"].shape)
```

同步策略：

- 每段音频按 `segment-sec` 切片。
- 对每段音频，使用该段的结束时间戳作为同步时间。
- 图像和 odom 会选择不晚于该同步时间、且最接近该同步时间的数据。
- `--max-diff-sec` 控制允许的最大时间差，默认 `0.2` 秒。

## 一步式：Bag + WAV 直接生成同步样本

如果不想手动先提取 bag，可以直接运行一步式脚本：

```bash
python data_processing/sync_from_bag_and_audio.py \
  --bag bags/bag_001 \
  --audio audio/clock1_1778111042843891356.wav \
  --output synced_dataset/bag_001 \
  --segment-sec 1.0 \
  --hop-sec 1.0
```

它会自动完成：

```text
ROS2 bag -> 临时提取图像/odom -> 按音频切段 -> 匹配最近图像/odom -> 生成 000001/000002/...
```

如果想保留中间提取文件：

```bash
python data_processing/sync_from_bag_and_audio.py \
  --bag bags/bag_001 \
  --audio audio/clock1_1778111042843891356.wav \
  --output synced_dataset/bag_001 \
  --keep-extracted
```

只使用部分 topic：

```bash
python data_processing/sync_from_bag_and_audio.py \
  --bag bags/bag_001 \
  --audio audio/clock1_1778111042843891356.wav \
  --output synced_dataset/bag_001 \
  --topics color,depth,lio_robo_odom
```

## 批量一步式同步

如果有多个 bag 和多个 wav，可以把它们分别放在两个目录。脚本会按排序顺序一一配对：

```text
bags/
├── bag_001/
├── bag_002/
└── bag_003/

audio/
├── clock1_1778111042843891356.wav
├── clock1_1778111052843891356.wav
└── clock1_1778111062843891356.wav
```

运行：

```bash
python data_processing/sync_from_bag_and_audio.py \
  --bag bags \
  --audio audio \
  --output synced_dataset \
  --topics color,depth,lio_robo_odom \
  --segment-sec 1.0 \
  --hop-sec 1.0
```

如果 bag 或 wav 在多层目录中：

```bash
python data_processing/sync_from_bag_and_audio.py \
  --bag bags \
  --audio audio \
  --output synced_dataset \
  --recursive
```

## 自定义 Topic

如果 topic 名字不同，可以手动指定：

```bash
python data_processing/extract_ros2_bag_data.py \
  --input bags \
  --output extracted_data \
  --color-topic /camera/color/image_raw \
  --depth-topic /camera/depth/image_raw \
  --lio-odom-topic /lio/odom \
  --lio-robo-odom-topic /lio/robo/odom \
  --livox-topic /livox/lidar
```
