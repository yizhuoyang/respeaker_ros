# ROS2 Bag Data Extraction

这个目录用于把 ROS2 bag 中的图像、里程计和 Livox 点云提取到普通文件夹。

脚本：

```text
extract_ros2_bag_data.py
sync_audio_with_extracted_data.py
sync_from_bag_and_audio.py
inspect_processed_data.ipynb
truncate_synced_dataset.py
compute_doa_from_odom.py
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

图像 topic 同时支持两种消息：

```text
sensor_msgs/msg/Image
sensor_msgs/msg/CompressedImage
```

如果你指定的是 raw topic，例如：

```text
/camera/color/image_raw
```

但 bag 里实际只有：

```text
/camera/color/image_raw/compressed
```

脚本会自动尝试读取 `/compressed`。深度图也会自动尝试 `/compressedDepth`。

compressed 图像会优先解码保存为 `.png`；如果缺少 OpenCV 或无法解码，会保存原始压缩数据，例如 `.jpg`、`.png` 或 `.bin`。

## 检查处理后的数据

处理完成后，可以打开 notebook 检查 odom 轨迹：

```text
data_processing/inspect_processed_data.ipynb
```

主要功能：

- 读取 `lio_odom.npz` / `lio_robo_odom.npz`
- 画机器人 XY 平面轨迹
- 画 XYZ 三维轨迹
- 查看 `x,y,z` 随样本编号变化
- 检查同步误差 `diff_ns`
- 检查 audio / image / odom / metadata 文件数量

打开后只需要修改：

```python
DATASET_DIR = Path("synced_dataset/bag_001")
```

## 裁剪同步后的数据集

如果同步后的数据太长，想只保留某个 index 之前的数据，可以使用：

```bash
python data_processing/truncate_synced_dataset.py \
  --dataset synced_dataset/bag_001 \
  --keep-through 120
```

这会保留：

```text
000000 到 000120
```

并删除之后的样本文件。

建议先 dry-run 看看会删什么：

```bash
python data_processing/truncate_synced_dataset.py \
  --dataset synced_dataset/bag_001 \
  --keep-through 120 \
  --dry-run
```

脚本会同步更新：

- `dataset_manifest.json`
- `lio_odom.npz`
- `lio_robo_odom.npz`
- `audio/ color/ depth/ lio_odom/ lio_robo_odom/ metadata/` 下对应 index 之后的文件

## 由 Odom 计算 DOA

如果把轨迹最终位置当作发声物体位置，可以计算每个时刻机器人头方向和目标方向在水平面上的夹角：

```bash
python data_processing/compute_doa_from_odom.py \
  --dataset synced_dataset/bag_001 \
  --odom lio_robo_odom
```

默认 target 是 odom 的最终位置。也可以手动指定发声物体位置：

```bash
python data_processing/compute_doa_from_odom.py \
  --dataset synced_dataset/bag_001 \
  --odom lio_robo_odom \
  --target 1.2,3.4,0.5
```

如果使用的是 `lio_odom.npz`，并且 odom 位姿代表的是 LiDAR 坐标系，但你希望按麦克风/机器人头的位置和朝向计算 DOA，可以先应用 LiDAR 到麦克风的外参：

```bash
python data_processing/compute_doa_from_odom.py \
  --dataset synced_dataset/bag_001 \
  --odom lio_odom \
  --input-frame lidar \
  --lidar-pitch-deg -23 \
  --mic-translation 0.2,0.0,0.0
```

这里的约定是：

- `--lidar-pitch-deg -23` 表示 LiDAR 相对麦克风/机器人头坐标系向下倾斜 23 度
- `--mic-translation 0.2,0.0,0.0` 表示麦克风原点相对 LiDAR 原点的平移，单位是米，坐标表达在 LiDAR 坐标系下
- 如果实际安装是麦克风在 LiDAR 后方、左侧或上方，需要按真实方向修改这个三维平移，例如 `-0.2,0,0` 或 `0,0,0.2`

输出：

```text
doa_lio_robo_odom.csv
doa_lio_robo_odom.npz
doa_lio_robo_odom/
├── 000000.npy
├── 000001.npy
└── ...
distance_lio_robo_odom.csv
distance_lio_robo_odom.npz
distance_lio_robo_odom/
├── 000000.npy
├── 000001.npy
└── ...
```

其中 `doa_lio_robo_odom/` 是逐样本保存的 DOA 特征目录，文件编号和同步数据中的 `audio/ color/ depth/ lio_odom/` 等目录保持一致。每个 `.npy` 内部是一维数组，字段顺序可以从同目录下的 `doa_lio_robo_odom.npz` 里的 `fields` 读取。

`distance_lio_robo_odom/` 是逐样本保存的距离目录。每个 `.npy` 内部包含：

```text
distance_xy
distance_3d
```

其中 `distance_xy` 是只看水平 `x-y` 平面的距离，`distance_3d` 是包含 `z` 的三维距离。训练水平 DOA 或平面导航任务时，通常优先使用 `distance_xy`。

如果想自定义逐样本输出目录名：

```bash
python data_processing/compute_doa_from_odom.py \
  --dataset synced_dataset/bag_001 \
  --odom lio_odom \
  --sample-output-dir doa
```

如果 `synced_dataset/` 下有多个已经同步好的数据集，可以批量处理：

```bash
python data_processing/compute_doa_from_odom.py \
  --dataset synced_dataset \
  --odom lio_odom \
  --input-frame lidar \
  --lidar-pitch-deg -23 \
  --mic-translation -0.2,0.0,0.0
```

如果数据集在更深层目录，增加：

```bash
--recursive
```

其中包含：

- robot 当前水平位置 `robot_x, robot_y`
- 原始 odom 水平位置 `source_odom_x, source_odom_y`
- target 水平位置 `target_x, target_y`
- 目标相对机器人在 world 坐标系下的水平向量 `target_vector_world_x, target_vector_world_y`
- 机器人头方向，也就是 body `+x` 轴投影到 world `x-y` 平面后的方向 `heading_world_x, heading_world_y`
- 到 target 的水平距离 `distance_xy`
- 到 target 的三维距离 `distance_3d`
- 机器人头方向在 world 水平面的角度 `robot_heading_world_deg`
- 目标方向在 world 水平面的角度 `target_azimuth_world_deg`
- 机器人头方向到目标方向的水平有符号夹角 `heading_target_yaw_signed_deg`
- 机器人头方向到目标方向的水平绝对夹角 `heading_target_yaw_abs_deg`

角度只使用 `x-y` 平面，不使用 `z`。在 ROS 常用坐标约定下，机器人 body `+x` 是头/前方，body `+y` 是左侧，所以 `heading_target_yaw_signed_deg` 为正通常表示目标在机器人头方向左侧，为负表示在右侧。

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
