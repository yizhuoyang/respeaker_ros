# SSLNet ROS1 实时音频推理与可视化

总流程入口见仓库根目录 [`README.md`](../README.md)；RGB-D YOLOE visual map 与
audio/visual/LiDAR 联合叠加见 [`deploy_yolo/README.md`](../deploy_yolo/README.md)。

你提到的 `ros_infer` 功能在当前仓库中实现为 `deploy/` 目录。本目录将已经训练好的
`SSLNet_DOA` 音频模型连接到 ROS1 的实时 ReSpeaker 音频流，输出目标方向
（DOA）和距离（distance）的概率分布以及峰值预测。

## 目录内容

```text
deploy/
├── sslnet_realtime.py          # 与 ROS 无关的模型加载、特征提取、滑动窗口推理核心
├── ros1_sslnet_audio_node.py   # ROS1 订阅音频并发布预测结果的节点
├── export_sslnet_tensorrt.py   # 将 audio-only .pth 导出为 ONNX/TensorRT engine
├── sslnet_engine_realtime.py   # TensorRT engine 推理核心
├── ros1_sslnet_audio_engine_node.py # 使用 engine 发布相同 ROS 推理 topic
├── ros1_sslnet_visualizer.py   # ROS1 在线 matplotlib 可视化节点
├── ros1_sslnet_rviz_markers.py # 在 RViz/LiDAR 点云中叠加预测 marker
├── ros1_livox_custom_to_pointcloud2.py # Livox CustomMsg 转 RViz 点云
├── ros1_sslnet_audio_map_fusion.py # 结合 odom 维护全局 audio map
├── ros1_sslnet_fake_prediction.py # 无模型时模拟 noisy DOA/distance 输出
└── README.md                    # 本使用说明
```

各部分的职责如下：

| 文件 | 用途 | 何时使用 |
| --- | --- | --- |
| `sslnet_realtime.py` | 从 checkpoint 恢复模型及训练预处理参数；将一段多通道音频转换为特征；输出 DOA/distance 概率 | 编写新的部署应用或不通过 ROS 做调用时 |
| `ros1_sslnet_audio_node.py` | 累积实时音频片段，按窗口触发推理并发布 ROS topic | 在线推理时必须运行 |
| `export_sslnet_tensorrt.py` | 从 audio-only checkpoint 导出 ONNX、TensorRT engine 及预处理 metadata | 使用 TensorRT 加速部署前运行一次 |
| `ros1_sslnet_audio_engine_node.py` | 与 `.pth` 节点发布相同 topic，但网络前向由 TensorRT engine 执行 | 已导出 engine 后替代 `.pth` 节点 |
| `ros1_sslnet_visualizer.py` | 订阅推理输出，以极坐标、概率曲线和俯视图实时展示结果 | 需要观察模型输出时运行 |
| `ros1_sslnet_rviz_markers.py` | 将预测转为 RViz 箭头、点、距离圆及分布 marker | 与 LiDAR 点云叠加验证时运行 |
| `ros1_livox_custom_to_pointcloud2.py` | 将运行中的 Livox `CustomMsg` 转为 `PointCloud2` | 不能重启 LiDAR 驱动时运行 |
| `ros1_sslnet_audio_map_fusion.py` | 使用 odom 与每帧分布累计全局 audio map，发布全局声源 argmax | 机器人移动时定位持续声源 |
| `ros1_sslnet_fake_prediction.py` | 以轨迹最后一个位置为假声源，生成带噪声的 DOA/distance 分布 | 没有训练模型时验证融合流程 |

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
  -> ros1_sslnet_audio_map_fusion.py 将多帧预测融合到 odom 固定坐标系
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

### 使用 TensorRT engine 替代 `.pth` 前向

TensorRT 版本仍在 CPU 侧执行与训练一致的音频预处理与 IPD 特征提取，只将
`SSLNet_DOA` 网络前向从 PyTorch 替换为 engine。它发布与
`ros1_sslnet_audio_node.py` 完全相同的 `/sslnet_audio_inference/*` topic，
所以后续的 visualizer、RViz marker 与 `ros1_sslnet_audio_map_fusion.py` 不需要修改。

在包含 CUDA、`onnx` 和 Python `tensorrt` 的 TensorRT 环境中，将当前权重导出为
FP32 engine。FP32 更适合首先验证 DOA 峰值与 `.pth` 一致；脚本可直接使用 TensorRT
Python Builder，如已安装 `trtexec` 也可使用该可执行程序：

```bash
cd /home/kemove/yyz/audio-nav/respeaker_ros

python deploy/export_sslnet_tensorrt.py \
  --checkpoint weights/pairs_ros1_sslnet_audio/best_model.pth \
  --onnx weights/pairs_ros1_sslnet_audio/best_model.onnx \
  --engine weights/pairs_ros1_sslnet_audio/best_model.engine \
  --window-seconds 1.0 \
  --device cuda:0 \
  --builder python
```

导出结果包括：

```text
weights/pairs_ros1_sslnet_audio/best_model.onnx
weights/pairs_ros1_sslnet_audio/best_model.engine
weights/pairs_ros1_sslnet_audio/best_model.engine.json
```

`.engine.json` 保存 `audio_feat`、麦克风通道、IPD pairs、滤波参数与固定输入 shape，
运行 engine 节点时必须与 `.engine` 一同保留。当前 `best_model.pth` 使用的 1 秒输入
shape 为 `(1, 12, 257, 101)`，输出为 `360` 维 DOA logits 与 `120` 维 distance logits。

如果 FP32 输出核对无误、并且更关注速度，可另导出 FP16 版本：

```bash
python deploy/export_sslnet_tensorrt.py \
  --checkpoint weights/pairs_ros1_sslnet_audio/best_model.pth \
  --onnx weights/pairs_ros1_sslnet_audio/best_model_fp16.onnx \
  --engine weights/pairs_ros1_sslnet_audio/best_model_fp16.engine \
  --window-seconds 1.0 \
  --device cuda:0 \
  --builder python \
  --fp16


python deploy/export_sslnet_tensorrt.py \
  --checkpoint weights/av_nav_sslnet_audio_more/best_model.pth \
  --onnx weights/av_nav_sslnet_audio_more/best_model.onnx \
  --engine weights/av_nav_sslnet_audio_more/best_model.engine \
  --window-seconds 0.5 \
  --device cuda:0 \
  --builder python \
```

FP16 可能在接近持平的 DOA bins 之间改变 argmax，使用前应在真实录音上与 FP32 或
`.pth` 对比误差。

启动 TensorRT 实时推理：

```bash
python deploy/ros1_sslnet_audio_engine_node.py \
  _engine:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.engine \
  _device:=cuda:0 \
  _window_seconds:=1.0 \
  _hop_seconds:=0.5



python deploy/ros1_sslnet_audio_engine_node.py \
  _engine:=weights/av_nav_sslnet_audio_more/best_model.engine \
  _device:=cuda:0 \
  _window_seconds:=0.5 \
  _hop_seconds:=0.25
```

节点已有相同路径的默认 engine，因此导出到上述位置后也可简写为：

```bash
python deploy/ros1_sslnet_audio_engine_node.py _hop_seconds:=0.5
```

engine 输入 shape 与导出窗口绑定：若希望用 `0.5 s` 输入，需要以
`--window-seconds 0.5` 另行导出 engine；`_hop_seconds` 仅控制更新频率，可以小于
导出窗口。

随后仍直接启动同一个 audio map 融合节点：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py
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

## 6. 维护全局 Audio Map

单帧的红色预测点会随着音频噪声波动。`ros1_sslnet_audio_map_fusion.py` 订阅每一帧
SSLNet 的 DOA/distance 分布和机器人 odom，并使用
`utlis/prob_update_doa.py` 中的 `StreamingSourceMapFusion` 在 odom 固定坐标系中累计
全局声源概率图。它会显式转换实时模型的坐标约定：

```text
SSLNet: 0 deg=机器人 +x/front, 90 deg=机器人 +y/left
ROS map: x/y 平面与 odom yaw
```

因此发布的全局声源 argmax 可直接与 ROS odom 地图或已变换到 odom frame 的 LiDAR 点云
叠加。

默认实时配置订阅 `/Odometry`，并使用与 visual map 一致的固定地图中心 `(0.0, 0.0)`、
`12.0 m x 12.0 m` 地图和 `0.05 m/cell` 分辨率。在音频推理节点与 odom topic
已经运行时直接启动：

```bash
source /opt/ros/noetic/setup.bash
cd /home/kemove/yyz/audio-nav/respeaker_ros

python deploy/ros1_sslnet_audio_map_fusion.py
```

发布 topic：

```text
/sslnet_audio_map/map
  nav_msgs/OccupancyGrid，全局 audio map 热力图

/sslnet_audio_map/markers
  visualization_msgs/MarkerArray，彩色格子为 heatmap，红点为全局 argmax，蓝箭头为当前机器人位姿，蓝线为运动轨迹

/sslnet_audio_map/argmax
  geometry_msgs/PointStamped，全局预测声源点

/sslnet_audio_map/argmax_json
  String，包含 x/y、frame_id、是否更新和置信度

/sslnet_audio_map/status
  String，即使尚未成功融合也会发布，指出当前仍在等待的输入
```

在 RViz 中进行全局融合验证：

1. 将 `Fixed Frame` 设置为 odom 消息的 `header.frame_id`，例如 `odom` 或 `camera_init`。
2. 添加 `MarkerArray`，topic 选择 `/sslnet_audio_map/markers`。其中彩色格子 heatmap 与红点
   使用完全相同的 marker 坐标链，适合与 LiDAR 对齐验证。
3. `/sslnet_audio_map/map` 保留为 `OccupancyGrid` 输出；如 RViz 的 `Map` 显示位置与 marker
   不一致，请关闭 `Map` 显示，直接使用上一步的彩色 marker heatmap。
4. 添加已处于相同固定坐标系、或有 TF 可变换到该坐标系的 LiDAR `PointCloud2`。

这样一张 RViz 图中会同时显示：

```text
LiDAR 点云                 环境与目标几何位置
彩色格子 heatmap           多帧融合概率，红色越深概率越高
红色球点                   当前全局预测声源 argmax
蓝色箭头 / 蓝色轨迹线      当前机器人朝向 / 已经过的位置
绿色球点（fake 模式）      用于比较的真实模拟声源位置
```

机器人轨迹最多保存最近 `1000` 次融合位置，可通过参数修改：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py \
  _odom_topic:=/Odometry \
  _robot_path_length:=3000
```

注意：全局 map 位于 odom 固定坐标系，不能在机器人运动时简单固定显示到
`livox_frame`。点云仍使用 `livox_frame` 时，RViz 必须能获得 odom frame 到
`livox_frame` 的实时 TF。

`_broadcast_odom_tf` 不是通常需要打开的选项。只有确认 TF 树中没有这一条变换，并且
`/Odometry` 的 pose 确实就是 `livox_frame` 在全局 frame 下的位姿时，才可以测试由
audio map 节点广播 TF：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py \
  _broadcast_odom_tf:=true \
  _sensor_frame_id:=livox_frame
```

如果开启后位置变差，应立即去掉 `_broadcast_odom_tf:=true` 与 `_sensor_frame_id`，
继续使用 LIO 或机器人系统已有的 TF。如果 odom 表示机器人机体而 LiDAR 与机体之间
存在外参，也必须使用已有的 TF 树或正确配置外参。

### 在 `livox_frame` 中同时观察点云与 Audio Map

如果希望 RViz 的观察坐标统一显示为 LiDAR 当前坐标系，应让 audio map 继续发布在
`/Odometry/header.frame_id` 给出的全局 frame 下，并将 RViz 的 `Fixed Frame` 改为
`livox_frame`。RViz 会通过 TF 将历史 audio map、声源点和机器人轨迹转换到当前
LiDAR 视角；原始 LiDAR 点云本身已经位于 `livox_frame`。

首先确认两个 frame：

```bash
rostopic echo -n 1 /Odometry/header.frame_id
rostopic echo -n 1 /livox/points_rviz/header/frame_id
```

推荐保持现有能够正确显示 LiDAR 与机器人 marker 的 TF 配置，启动 audio map 时不额外
广播 TF：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py
```

在 RViz 中沿用当前显示正确的 `Fixed Frame`，并使用 marker heatmap：

```text
Fixed Frame: 当前已能正确显示点云和机器人 marker 的 frame
MarkerArray: /sslnet_audio_map/markers
PointCloud2: /livox/points_rviz
```

不要对 audio map 节点设置 `_frame_id:=livox_frame`。`_frame_id` 仅在 odom 消息缺少
`header.frame_id` 时作为后备的全局 frame；如果它与 odom 的 frame 冲突，节点会忽略
该参数并给出警告。已经以错误 frame 启动过节点时，请重启节点或调用 reset 服务清空
此前累计的 heatmap：

```bash
rosservice call /sslnet_audio_map/reset
```

彩色 marker heatmap 的显示参数：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py \
  _heatmap_marker_threshold:=0.08 \
  _heatmap_marker_max_cells:=6000 \
  _heatmap_marker_height:=0.02
```

- `_publish_heatmap_marker`：是否在 `/sslnet_audio_map/markers` 中发布彩色格子 heatmap，默认 `true`。
- `_heatmap_marker_threshold`：仅显示相对于当前峰值高于该比例的格子，默认 `0.08`。
- `_heatmap_marker_max_cells`：最多发布的彩色格子数，默认 `6000`。
- `_heatmap_marker_height`：heatmap 在 RViz 平面上方的高度，默认 `0.02 m`。

如果 `/sslnet_audio_map/markers` 没有输出，请注意该 topic 只有在首次成功融合之后才会
产生消息。先确认已重启到最新节点，再查看诊断状态：

```bash
rostopic echo -n 1 /sslnet_audio_map/status
```

`waiting_for` 的含义：

```text
odom                              未收到默认 /Odometry 或指定的 odom topic
prediction_json                   未收到真实或 fake 推理摘要
doa_distribution                  未收到 DOA 分布
distance_distribution             未收到 distance 分布
odom_at_or_after_prediction_stamp 预测时间晚于当前最新 odom
odom_timestamp_difference_too_large 预测与最近 odom 相差超过 `_max_odom_diff_sec`
new_prediction_distribution       当前预测已处理，等待下一帧
```

确认各输入确实存在：

```bash
rostopic hz /Odometry
rostopic hz /sslnet_audio_inference/prediction_json
rostopic hz /sslnet_audio_inference/doa_distribution
rostopic hz /sslnet_audio_inference/distance_distribution
```

常用融合参数：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py \
  _map_size_m:=12.0 \
  _resolution:=0.05 \
  _beta_r:=0.05 \
  _sigma_Q_cells:=1.5 \
  _min_confidence:=0.05 \
  _max_audio_input_mean_abs:=0.6
```

- `_map_size_m` 和 `_resolution`：全局方形地图边长与格子分辨率。
- `_map_center_x` 和 `_map_center_y`：固定全局地图中心，默认均为 `0.0`；与 visual map
  的默认值一致，两张地图不需要互相订阅也能严格逐格叠加。
- `_beta_r`：distance 分布在融合中的影响，默认 `0.05`；越大越依赖距离预测，设为 `0`
  则只使用 DOA。
- `_sigma_Q_cells`：每次更新前对历史 map 做空间扩散，默认 `1.5`。它不是严格的
  temporal decay，但能避免历史峰值过尖导致后续帧很难更新。
- `_min_confidence`：DOA 峰值置信度低于该阈值时跳过当前更新。`doa_confidence`
  来自 softmax 后 DOA 分布的最大概率，范围为 `0~1`。
- `_max_audio_input_mean_abs`：网络输入音频通道 `1,2,3,4` 的平均绝对幅值高于该阈值时跳过
  当前 map 更新，默认 `0.6`；设为 `0` 或负数可关闭该规则。
- `_argmax_resolution`：输出声源位置的网格精度，默认与 `_resolution` 相同。
- `_max_odom_diff_sec`：预测时间戳可匹配的最大 odom 时间差，默认 `0.20 s`。

切换到新的声源目标或重新开始实验时，清空已经累计的 map：

```bash
rosservice call /sslnet_audio_map/reset
```

## 7. 没有训练模型时验证 Audio Map

`ros1_sslnet_fake_prediction.py` 可以替代真实 SSLNet 推理节点。它将轨迹最后一帧 odom 的
`x/y` 位置视为一个固定声源；随后每接收一帧当前 odom，就计算机器人相对该声源的真实
DOA/distance，再为角度和距离加入高斯噪声并发布模拟概率分布：

```text
当前 odom + 轨迹最终 odom 位置
  -> ground-truth DOA/distance
  -> 加入角度/距离噪声
  -> 发布 /sslnet_audio_inference/* 假预测
  -> ros1_sslnet_audio_map_fusion.py 融合并显示全局 argmax
```

它发布的 topic 与真实网络完全相同，因此使用 fake 节点时不要同时启动
`ros1_sslnet_audio_node.py`。

### 使用 ROS1 `.bag` 取最后一帧声源位置

先确认 bag 中的 odom topic：

```bash
rosbag info /path/to/test.bag | grep -E "/lio/odom|/lio/robo/odom"
```

终端 1：启动 fake prediction 节点。节点会先扫描 bag 的最后一帧 `/lio/odom`，保存其
位置作为绿色 ground-truth 声源点：

```bash
source /opt/ros/noetic/setup.bash
cd /home/kemove/yyz/audio-nav/respeaker_ros

python deploy/ros1_sslnet_fake_prediction.py \
  _bag_path:=/path/to/test.bag \
  _bag_odom_topic:=/lio/odom \
  _odom_topic:=/lio/odom \
  _angle_noise_std_deg:=8.0 \
  _distance_noise_std_m:=0.15
```

终端 2：启动全局融合：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py \
  _odom_topic:=/lio/odom \
  _map_size_m:=12.0 \
  _resolution:=0.05 \
  _max_distance_m:=6.0 \
  _min_confidence:=0.0
```

终端 3：播放同一个 bag：

```bash
rosparam set use_sim_time true
rosbag play --clock /path/to/test.bag
```

### 使用已提取的 odom NPZ 取声源位置

若原始数据来自 ROS2 bag，或已通过数据处理流程生成
`lio_odom.npz`，fake 节点可以直接读取 NPZ 最后一行作为声源位置，同时订阅正在回放或
实时发布的同坐标系 odom topic：

```bash
python deploy/ros1_sslnet_fake_prediction.py \
  _odom_npz:=/path/to/sequence/lio_odom.npz \
  _odom_topic:=/lio/odom \
  _angle_noise_std_deg:=8.0 \
  _distance_noise_std_m:=0.15
```

也可以跳过 bag，手动指定一个固定声源世界坐标：

```bash
python deploy/ros1_sslnet_fake_prediction.py \
  _source_x:=3.0 \
  _source_y:=-1.5 \
  _odom_topic:=/lio/odom
```

### RViz 中对比 Ground Truth 与融合结果

在前述全局 RViz 配置基础上，再添加一个 `MarkerArray`：

```text
/sslnet_fake_prediction/markers
```

显示含义：

```text
绿色球点    fake 数据使用的真实固定声源位置，即轨迹最后一帧位置
红色球点    audio map 当前融合后的 argmax 预测位置
热力图      累积后的全局声源概率
蓝色箭头    当前 odom 中的机器人位姿
```

可查看每帧模拟预测及其误差：

```bash
rostopic echo /sslnet_audio_inference/prediction_json
```

其中额外包含：

```text
ground_truth_source_x / ground_truth_source_y
ground_truth_doa_deg / ground_truth_distance_m
doa_abs_error_deg / distance_abs_error_m
```

常用模拟参数：

```bash
python deploy/ros1_sslnet_fake_prediction.py \
  _bag_path:=/path/to/test.bag \
  _angle_noise_std_deg:=12.0 \
  _distance_noise_std_m:=0.25 \
  _doa_sigma_deg:=8.0 \
  _distance_sigma_m:=0.20 \
  _uniform_noise_weight:=0.05 \
  _seed:=0
```

- `_angle_noise_std_deg` 与 `_distance_noise_std_m`：每帧峰值预测的随机误差。
- `_doa_sigma_deg` 与 `_distance_sigma_m`：输出概率分布本身的展宽。
- `_uniform_noise_weight`：混入均匀背景概率的比例。
- `_publish_hz`：限制 fake 输出频率；默认 `0` 表示每条 odom 都输出一帧。
- `_seed`：固定随机种子，便于复现实验。

如果某些帧到最终声源位置的距离超过 `_max_distance_m`，模拟距离会被裁剪，融合验证不再
等价于完整真实距离。此时应增大 fake 节点和 audio map 节点两侧一致的
`_max_distance_m`。

## 8. 与 YOLOE Visual Map 联合运行

`deploy_yolo/ros1_yoloe_visual_map.py` 会将 RGB-D 检测目标累计为全局视觉占据图。两张
地图默认都使用 `12.0 m x 12.0 m` 与 `0.05 m/cell`，默认固定中心也均为
`(0.0, 0.0)`，当前实时配置可直接启动并叠加：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py

python deploy_yolo/ros1_yoloe_visual_map.py
```

其中 visual map 默认加载 `deploy_yolo/yoloe-11s-seg.engine`，订阅注册深度
`/camera/depth/image_rect_raw`，采用 `_conf:=0.50` 与 `_max_depth_m:=4.0`。
若回放 `/lio/odom` 的 bag，请在两个节点均显式追加 `_odom_topic:=/lio/odom`。

在 RViz 中添加：

```text
MarkerArray: /sslnet_audio_map/markers
MarkerArray: /yoloe_visual_map/markers
PointCloud2: /livox/points_rviz
```

visual map 的相机内参、对齐深度、TensorRT engine、marker 与排障说明请查阅
[`deploy_yolo/README.md`](../deploy_yolo/README.md)。

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
