# SSLNet ROS1 实时音频推理与可视化

你提到的 `ros_infer` 功能在当前仓库中实现为 `deploy/` 目录。本目录将已经训练好的
`SSLNet_DOA` 音频模型连接到 ROS1 的实时 ReSpeaker 音频流，输出目标方向
（DOA）和距离（distance）的概率分布以及峰值预测。

## 目录内容

```text
deploy/
├── sslnet_realtime.py          # 与 ROS 无关的模型加载、特征提取、滑动窗口推理核心
├── ros1_sslnet_audio_node.py   # ROS1 订阅音频并发布预测结果的节点
├── ros1_sslnet_visualizer.py   # ROS1 在线 matplotlib 可视化节点
├── ros1_sslnet_rviz_markers.py # 在 RViz/LiDAR 点云中叠加预测 marker
├── ros1_livox_custom_to_pointcloud2.py # Livox CustomMsg 转 RViz 点云
└── README.md                    # 本使用说明
```

各部分的职责如下：

| 文件 | 用途 | 何时使用 |
| --- | --- | --- |
| `sslnet_realtime.py` | 从 checkpoint 恢复模型及训练预处理参数；将一段多通道音频转换为特征；输出 DOA/distance 概率 | 编写新的部署应用或不通过 ROS 做调用时 |
| `ros1_sslnet_audio_node.py` | 累积实时音频片段，按窗口触发推理并发布 ROS topic | 在线推理时必须运行 |
| `ros1_sslnet_visualizer.py` | 订阅推理输出，以极坐标、概率曲线和俯视图实时展示结果 | 需要观察模型输出时运行 |
| `ros1_sslnet_rviz_markers.py` | 将预测转为 RViz 箭头、点、距离圆及分布 marker | 与 LiDAR 点云叠加验证时运行 |
| `ros1_livox_custom_to_pointcloud2.py` | 将运行中的 Livox `CustomMsg` 转为 `PointCloud2` | 不能重启 LiDAR 驱动时运行 |

## 推理流程

在线处理链路是：

```text
ReSpeaker 麦克风
  -> respeaker_ros_recorder 发布 /respeaker/audio_raw
  -> ros1_sslnet_audio_node.py 累积 0.5s 或 1s 音频窗口
  -> 从 checkpoint 还原通道选择、IPD 特征和模型
  -> SSLNet_DOA 输出 360 维 DOA 分布与 120 维距离分布
  -> 发布 prediction / prediction_json / 两个完整分布 topic
  -> ros1_sslnet_visualizer.py 实时显示结果
  -> ros1_sslnet_rviz_markers.py 在 RViz 中与 LiDAR 点云叠加
```

`ros1_sslnet_audio_node.py` subscribes to live multichannel ReSpeaker audio and
publishes SSLNet audio-only DOA and distance predictions. It matches the
`pairs_ros1` training convention:

- DOA bin `0 deg`: LiDAR/robot `+x` (front).
- DOA bin `90 deg`: LiDAR/robot `+y` (left).
- Distance output: planar range from `0` to `6 m` over 120 bins.

## 坐标和输出语义

本节点与 `pairs_ros1` 的 bbox 标签保持同一个坐标约定：

- `0 deg`：机器人或 LiDAR 的 `+x` 方向，即正前方。
- `90 deg`：机器人或 LiDAR 的 `+y` 方向，即左侧。
- `180 deg`：后方。
- `270 deg`：右侧。
- distance 是平面距离，当前分布范围为 `0 m` 到 `6 m`，共 `120` 个 bin。

例如 `doa_deg=45`、`distance_m=2.0` 表示模型预测声源在机器人左前方约 `2 m`。

## 输入 Topic

推理节点订阅：

```text
/respeaker/audio_raw
respeaker_ros_recorder/AudioDataStamped
```

with fields:

```text
std_msgs/Header header
uint32 sample_rate
uint32 channels
uint32 frames
int16[] data
```

模型训练使用 `pairs_ros1` 的 `16000 Hz`、6 通道 wav，其中 checkpoint 会记录模型真正选用的
通道，例如本次权重使用 `1,2,3,4` 四个麦克风通道。

## 准备环境

开三个终端，并在每个终端中加载 ROS1 和 ReSpeaker 消息包：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/yyz/respeaker_ros/devel/setup.bash
cd /home/kemove/yyz/audio-nav/respeaker_ros
```

先确认权重存在：

```bash
ls weights/pairs_ros1_sslnet_audio/best_model.pth
```

## 1. 启动实时音频采集

在终端 1 运行：

```bash
roslaunch respeaker_ros_recorder respeaker_collect.launch
```

检查音频 topic 是否持续发布：

```bash
rostopic hz /respeaker/audio_raw
rostopic type /respeaker/audio_raw
```

预期消息类型为：

```text
respeaker_ros_recorder/AudioDataStamped
```

采集配置必须为 `16000 Hz`、6 通道，才能与训练数据的输入条件一致。

## 2. 启动实时推理

### 推荐：1 秒窗口，0.5 秒更新

如果训练数据是 1 秒音频片段，优先使用该配置。在终端 2 运行：

```bash
python deploy/ros1_sslnet_audio_node.py \
  _checkpoint:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.pth \
  _device:=cuda:0 \
  _window_seconds:=1.0 \
  _hop_seconds:=0.5
```

含义：

- `_window_seconds:=1.0`：每次推理观察最近 1 秒音频。
- `_hop_seconds:=0.5`：每收到新的 0.5 秒内容，产生一次新预测。
- 推理窗口会重叠，因此方向曲线通常比非重叠窗口更平滑。
- 音频回调只提交最新窗口，模型推理由后台线程执行；如果 GPU 推理暂时变慢，节点会丢弃旧窗口而不是积压延迟。

### 低延迟：0.5 秒窗口

```bash
python deploy/ros1_sslnet_audio_node.py \
  _checkpoint:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.pth \
  _device:=cuda:0 \
  _window_seconds:=0.5 \
  _hop_seconds:=0.5
```

由于模型中使用自适应池化，0.5 秒输入可以运行。不过，如果当前模型是用 1 秒片段训练的，
0.5 秒时的性能可能下降。长期使用低延迟模式时，建议重新使用 0.5 秒片段训练或微调模型。

### CPU 运行

没有 CUDA 环境时可直接使用：

```bash
python deploy/ros1_sslnet_audio_node.py \
  _checkpoint:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.pth \
  _device:=cpu \
  _window_seconds:=1.0 \
  _hop_seconds:=0.5
```

### 正常停止节点

在运行推理节点的终端按：

```text
Ctrl+C
```

节点关闭时会立刻取消音频订阅、清空尚未推理的窗口，并仅短暂等待正在运行的一次推理结束。
若 CUDA 调用卡住，后台推理线程不会无限阻塞 ROS 关闭。

可调整关闭等待时间和待处理窗口队列大小：

```bash
python deploy/ros1_sslnet_audio_node.py \
  _checkpoint:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.pth \
  _device:=cuda:0 \
  _window_seconds:=1.0 \
  _hop_seconds:=0.5 \
  _shutdown_timeout:=0.5 \
  _inference_queue_size:=1
```

其中：

- `_shutdown_timeout`：关闭时等待后台推理线程的最长秒数，默认 `0.5`。
- `_inference_queue_size`：尚未处理的窗口数量上限，默认 `1`；实时导航通常应保持为 `1`，优先使用最新声音观测。

如果旧版本节点已经卡死，或进程在更新代码前已经启动，可从另一终端结束它：

```bash
pkill -TERM -f "deploy/ros1_sslnet_audio_node.py"
sleep 1
pkill -KILL -f "deploy/ros1_sslnet_audio_node.py"
```

## 3. 查看数值结果

快速查看可读的 JSON 输出：

```bash
rostopic echo /sslnet_audio_inference/prediction_json
```

你会看到类似：

```json
{
  "doa_deg": 82.0,
  "distance_m": 3.08,
  "doa_confidence": 0.12,
  "distance_confidence": 0.18,
  "inference_ms": 8.4,
  "window_seconds": 1.0,
  "stamp": 1779578820.1
}
```

发布的全部 topic 如下：

```text
/sslnet_audio_inference/prediction
  Float32MultiArray [doa_deg, distance_m, doa_confidence,
                     distance_confidence, inference_ms]

/sslnet_audio_inference/prediction_json
  String with named values for logging and debugging

/sslnet_audio_inference/doa_distribution
  Float32MultiArray with 360 probability values

/sslnet_audio_inference/distance_distribution
  Float32MultiArray with 120 probability values
```

`prediction` 更适合给其他 ROS 节点消费，数组字段顺序固定为：

```text
[doa_deg, distance_m, doa_confidence, distance_confidence, inference_ms]
```

完整分布适合画图或做后续概率融合。DOA 第 `i` 个元素代表 `i deg` 的概率；distance 第
`i` 个元素对应：

```text
distance_m = i * 6 / 119
```

## 4. 实时图形可视化

确保推理节点已运行后，在终端 3 执行：

```bash
python deploy/ros1_sslnet_visualizer.py
```

窗口由三张图组成：

1. 左图 `DOA` 极坐标分布：蓝色曲线为 360 维方向概率，红线为当前峰值方向。
2. 中图 `Distance` 分布：蓝绿曲线为距离概率，红线为当前峰值距离。
3. 右图 `Top-down prediction`：黑色方块为机器人，红色箭头和点为预测目标位置。

可视化坐标中，上述右图的 `+x` 是前方、`+y` 是左侧，因此图形语义与训练标签一致。

在可视化窗口中按键：

```text
s    保存当前画面 PNG 截图
```

指定截图目录：

```bash
python deploy/ros1_sslnet_visualizer.py \
  _snapshot_dir:=$(pwd)/deploy_snapshots \
  _refresh_hz:=10
```

### 使用 `rqt_plot` 做简单曲线监视

仅关心峰值角度与距离随时间变化时，也可以运行：

```bash
rqt_plot /sslnet_audio_inference/prediction/data[0] \
         /sslnet_audio_inference/prediction/data[1]
```

其中 `data[0]` 是 DOA 度数，`data[1]` 是距离米数。角度在 `359 -> 0` 跨界时会表现为
跳变，这是角度环绕而不是模型瞬间转向；极坐标可视化更适合观察该情况。

## 5. 在 RViz 中叠加 LiDAR 验证

`ros1_sslnet_rviz_markers.py` 将音频预测投影到水平平面，发布
`/sslnet_rviz_markers/markers`。其中：

- 红色箭头和红点：当前 DOA 与 distance 的峰值预测位置。
- 绿色圆：预测峰值 distance 的等距圆。
- 黄色轮廓：归一化后的 DOA 概率分布，轮廓凸出的方向概率更高。
- 蓝色半透明圆环：distance 分布；颜色越明显，该半径的概率越高。

先确定 LiDAR 可以在 RViz 中使用的 topic 和坐标系：

```bash
rostopic list | grep -Ei "livox|lidar|points|scan"
rostopic type /livox/lidar
rostopic echo -n 1 /livox/lidar/header/frame_id
```

若 `/livox/lidar` 的类型是 `sensor_msgs/PointCloud2`，可在 RViz 中直接添加
`PointCloud2`。如果类型是 `livox_ros_driver2/CustomMsg`，有以下两种做法。

### 方案 A：让 Livox 驱动直接发布 `PointCloud2`（推荐）

你本机 `/home/kemove/driver_ws/src/livox_ros_driver2/launch_ROS1/msg_MID360.launch`
默认 `xfer_format=1`，会发布 `CustomMsg`。停止当前 LiDAR 驱动后，用
`xfer_format:=0` 重新启动：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/driver_ws/devel/setup.bash
roslaunch livox_ros_driver2 msg_MID360.launch xfer_format:=0
```

或直接使用驱动自带的 RViz 启动文件，它的默认值就是 `xfer_format=0`：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/driver_ws/devel/setup.bash
roslaunch livox_ros_driver2 rviz_MID360.launch
```

确认转换成功：

```bash
rostopic type /livox/lidar
rostopic echo -n 1 /livox/lidar/header/frame_id
```

预期第一个命令输出 `sensor_msgs/PointCloud2`，第二个命令可正常输出
`livox_frame` 等 frame 名称。

### 方案 B：保留 `CustomMsg`，额外转换为 RViz 点云

当 `/livox/lidar` 还要提供给其他需要 `CustomMsg` 的程序或录包流程时，不必重启驱动。
在新终端执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/driver_ws/devel/setup.bash
cd /home/kemove/yyz/audio-nav/respeaker_ros
python deploy/ros1_livox_custom_to_pointcloud2.py
```

转换后的点云 topic 为：

```text
/livox/points_rviz
sensor_msgs/PointCloud2
```

点云太密导致 RViz 卡顿时，可以只发布每 4 个点中的 1 个：

```bash
python deploy/ros1_livox_custom_to_pointcloud2.py _point_stride:=4
```

只保留高度不超过 `1.9 m` 的点：

```bash
python deploy/ros1_livox_custom_to_pointcloud2.py \
  _z_max:=1.9
```

也可以同时限制前后、左右、高度和水平距离范围。例如只显示机器人前方 `0` 到 `6 m`、
左右 `-3` 到 `3 m`、高度 `-0.4` 到 `1.9 m` 的点：

```bash
python deploy/ros1_livox_custom_to_pointcloud2.py \
  _x_min:=0.0 \
  _x_max:=6.0 \
  _y_min:=-3.0 \
  _y_max:=3.0 \
  _z_min:=-0.4 \
  _z_max:=1.9 \
  _range_max_m:=6.0
```

支持的过滤参数为 `_x_min/_x_max`、`_y_min/_y_max`、`_z_min/_z_max` 和
`_range_min_m/_range_max_m`；最后两个表示 `sqrt(x^2 + y^2)` 的水平距离。
所有参数都使用 LiDAR 的 `frame_id` 坐标，未指定的方向不进行裁剪。边界值会被保留，
因此 `_z_max:=1.9` 保留 `z <= 1.9` 的点。

你遇到的以下错误并不代表点云没有发布，而是运行 `rostopic` 的当前终端无法导入
Livox 自定义消息定义：

```text
ERROR: Cannot load message class for [livox_ros_driver2/CustomMsg]. Are your messages built?
```

加载驱动工作空间后即可查看 `CustomMsg`：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/driver_ws/devel/setup.bash
rostopic echo -n 1 /livox/lidar/header/frame_id
```

在推理节点正在输出预测时，另开终端启动 marker 节点。将 `livox_frame` 替换为上一步
读到的 LiDAR `header.frame_id`：

```bash
source /opt/ros/noetic/setup.bash
cd /home/kemove/yyz/audio-nav/respeaker_ros
python deploy/ros1_sslnet_rviz_markers.py \
  _frame_id:=livox_frame
```

然后启动 `rviz` 并配置：

1. 将 `Fixed Frame` 设置为 LiDAR 的 `header.frame_id`，例如 `livox_frame`。
2. 添加 `PointCloud2` 显示；方案 A 选择 `/livox/lidar`，方案 B 选择 `/livox/points_rviz`。
3. 添加 `MarkerArray` 显示，topic 选择 `/sslnet_rviz_markers/markers`。
4. 让真实声源在点云中可定位的位置发声，查看红点是否落在对应障碍物或人员附近。

如果麦克风原点不在 LiDAR 原点，或声学坐标方向相对点云存在偏角，可校准 marker：

```bash
python deploy/ros1_sslnet_rviz_markers.py \
  _frame_id:=livox_frame \
  _origin_x:=0.10 \
  _origin_y:=0.00 \
  _origin_z:=0.20 \
  _yaw_offset_deg:=0.0
```

`_origin_x/_origin_y/_origin_z` 表示麦克风阵列中心在 LiDAR frame 下的位置；
`_yaw_offset_deg` 为声学 `0 deg` 到 LiDAR `+x` 的旋转偏置，逆时针为正。模型训练约定
本来就是 `0 deg=+x/front`、`90 deg=+y/left` 时，该值应保持为 `0`。

还可关闭概率分布轮廓，仅保留峰值预测，以便观察密集点云：

```bash
python deploy/ros1_sslnet_rviz_markers.py \
  _frame_id:=livox_frame \
  _show_distributions:=false
```

## 参数覆盖与训练一致性

The checkpoint contains the training values of `audio_feat`, `audio_channels`,
`ipd_pairs`, compression, bandpass, and filter/mute preprocessing settings.
The node loads those values automatically. They can be overridden as private
ROS parameters, for example:

```bash
python deploy/ros1_sslnet_audio_node.py \
  _checkpoint:=/path/to/best_model.pth \
  _audio_channels:="0,1,2,3" \
  _audio_feat:=ipd \
  _ipd_pairs:="0-1,0-2,0-3,1-2,1-3,2-3"
```

The runtime channel selection must match the channel selection used for
training, or the learned spatial phase relationship will no longer align with
the live microphone array.

Checkpoints trained with `--use-filter-mute-denoise` are supported and restore
that preprocessing automatically. Checkpoints trained with `--use-denoise`
depend on offline noise-profile wav files and are rejected by this real-time
node until equivalent live noise-profile handling is configured.

尤其需要注意：

- 训练使用 `--audio-channels 1,2,3,4`，实时也必须使用相同的物理通道排序。
- 训练使用 `--audio-feat ipd`，部署不能任意改成 `spec` 或 `phase`。
- 只有通过 `--model audio` 训练的 checkpoint 可以用于当前纯音频实时节点；
  `audio_depth` 模型还需要同步订阅深度图像才能推理。
- `--use-filter-mute-denoise` 可以在线复现；旧的 `--use-denoise` 依赖离线噪声 profile，
  当前节点会拒绝不等价的实时部署。

## 常见排查

### 报错 `No module named 'respeaker_ros_recorder'`

`AudioDataStamped` 是 ReSpeaker catkin 工作空间编译生成的 ROS 消息类型。当前终端如果只
加载了 ROS，而没有加载 ReSpeaker 工作空间，Python 就找不到这个模块。启动采集、推理
或可视化之前，在对应终端执行：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/yyz/respeaker_ros/devel/setup.bash
python3 -c "from respeaker_ros_recorder.msg import AudioDataStamped; print('message import ok')"
```

节点也会尝试读取本机默认生成目录
`/home/kemove/yyz/respeaker_ros/devel/lib/python3/dist-packages`。如果消息包来自另一个
catkin 工作空间，可显式指定：

```bash
export RESPEAKER_ROS_MSG_PATH=/path/to/catkin_ws/devel/lib/python3/dist-packages
python deploy/ros1_sslnet_audio_node.py \
  _checkpoint:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.pth
```

### 没有输出预测

检查输入 topic 与频率：

```bash
rostopic list | grep audio
rostopic hz /respeaker/audio_raw
rostopic echo -n 1 /respeaker/audio_raw/sample_rate
```

第一次预测需要先积累满一个窗口。例如使用 `1.0s` 窗口时，节点启动后约 1 秒才开始输出。

### 报错 `Expected 16000 Hz audio`

实时采集采样率与训练不一致。将 ReSpeaker 采集节点设置为 `16000 Hz`，不要直接用不同
采样率测试已训练的模型。

### 报错通道数量不足

checkpoint 所需通道超出了实时消息通道数量。检查：

```bash
rostopic echo -n 1 /respeaker/audio_raw/channels
```

当前 `pairs_ros1` 模型期望采集消息提供 6 通道，并从中选择训练时记录的 4 个通道。

### 图形窗口打不开

`ros1_sslnet_visualizer.py` 需要桌面显示环境。如果通过 SSH 运行，请启用 X11 转发，
或只使用 `rostopic echo` / `rqt_plot` 查看数值输出。
