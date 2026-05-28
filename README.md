# Audio-Visual Pedestrian Aware Mapping

本仓库实现一条 ROS1 音频与视觉联合定位流程：

```text
pairs_ros1 数据
  -> SSLNet audio-only 训练与测试（DOA + distance 分布）
  -> ROS 实时音频预测或 fake prediction
  -> odom 融合得到全局 audio map 与声源 argmax

RGB + registered depth
  -> YOLOE segmentation/detection
  -> median-depth 目标区域投影
  -> 全局 visual map

audio map + visual map + LiDAR + robot trajectory
  -> RViz 中叠加验证声音位置与可见目标位置
```

## 文档导航

| 文档 | 内容 |
| --- | --- |
| [data_processing/README.md](data_processing/README.md) | ROS2 bag / 音频同步、DOA 与 distance 标签、噪声处理及数据检查 |
| [deploy/README.md](deploy/README.md) | SSLNet ROS1 实时推理、fake prediction、audio map、LiDAR 与 RViz |
| [deploy_yolo/README.md](deploy_yolo/README.md) | YOLOE RGB-D visual map、fake CameraInfo、TensorRT engine 与 RViz |

## 关键模块

```text
main_doa.py                         SSLNet DOA/distance 训练入口
test_doa.py                         SSLNet 测试、误差打印和分布导出
model_training/train_doa.py         训练/验证循环，打印 DOA peak MAE
dataloader/ssl_dataset.py           synced_dataset 与 pairs_ros1 数据读取

deploy/ros1_sslnet_audio_node.py    实时音频模型推理
deploy/export_sslnet_tensorrt.py    SSLNet checkpoint 导出 TensorRT engine
deploy/ros1_sslnet_audio_engine_node.py TensorRT 实时音频模型推理
deploy/ros1_sslnet_fake_prediction.py 无模型时生成模拟分布
deploy/ros1_sslnet_audio_map_fusion.py 全局 audio map 融合
deploy/ros1_livox_custom_to_pointcloud2.py Livox CustomMsg 转 RViz 点云

deploy_yolo/ros1_fake_camera_info.py 无标定信息时发布测试 CameraInfo
deploy_yolo/ros1_yoloe_visual_map.py YOLOE RGB-D 全局 visual map
```

## 坐标与地图约定

音频模型使用与 `pairs_ros1` 一致的平面方向约定：

```text
DOA 0 deg     机器人/LiDAR +x，正前方
DOA 90 deg    机器人/LiDAR +y，左侧
DOA 180 deg   后方
DOA 270 deg   右侧
distance      平面距离，默认 0 到 6 m，共 120 bins
```

两张全局地图的默认几何一致：

```text
map_size_m = 12.0
resolution = 0.05
```

实时节点当前默认使用同一个固定中心，因此直接启动即可逐格叠加：

```text
map_center_x = 0.0
map_center_y = 0.0
```

地图应发布在 odom 的全局 frame 中，例如 `camera_init` 或 `odom`，而不是直接将累计地图
重新标记为 `livox_frame`。LiDAR 点云通过 TF 叠加到全局地图。

实时部署的默认 odom topic 为 `/Odometry`。回放含 `/lio/odom` 的 bag 时，应在
fake prediction、audio map 与 visual map 命令中统一显式传入
`_odom_topic:=/lio/odom`，不要只修改其中一个节点。

## 1. 数据准备

已有数据路径为：

```text
/home/kemove/yyz/AV-PedAware/data/pairs_ros1
```

`main_doa.py` 和 `test_doa.py` 支持该数据集的 `train/test` 布局。若需要从 ROS2 bag
重新生成同步数据、计算 odom 标签或处理运动噪声，参见
[data_processing/README.md](data_processing/README.md)。

## 2. 训练 SSLNet Audio 模型

ROS 实时音频节点需要通过 `--model audio` 训练的 checkpoint；`audio_depth` checkpoint
不能直接用于纯音频实时节点。

推荐训练命令：

```bash
cd /home/kemove/yyz/audio-nav/respeaker_ros

python main_doa.py \
  --data-root /home/kemove/yyz/AV-PedAware/data/pairs_ros1 \
  --model audio \
  --audio-feat ipd \
  --audio-channels 1,2,3,4 \
  --ipd-pairs 0-1,0-2,0-3,1-2,1-3,2-3 \
  --allow-missing-depth \
  --epochs 80 \
  --batch-size 16 \
  --lr 1e-4 \
  --lr-scheduler cosine \
  --min-lr-ratio 0.05 \
  --noise-aug \
  --noise-aug-root /home/kemove/yyz/AV-PedAware/data/pairs_ros1/noise \
  --snr-min-db 0 \
  --snr-max-db 25 \
  --device cuda:0 \
  --save-dir weights/pairs_ros1_sslnet_audio \
  --log-dir runs/pairs_ros1_sslnet_audio
```

训练中每个 epoch 会打印：

```text
Epoch ... | lr=...
Train DOA peak MAE: ... deg
Val DOA peak MAE: ... deg
Train loss: ... | Val loss: ...
Next lr: ...
```

最优权重写入：

```text
weights/pairs_ros1_sslnet_audio/best_model.pth
```

测试并导出预测误差：

```bash
python test_doa.py \
  --data-root /home/kemove/yyz/AV-PedAware/data/pairs_ros1 \
  --eval-split val \
  --model audio \
  --audio-feat ipd \
  --audio-channels 1,2,3,4 \
  --ipd-pairs 0-1,0-2,0-3,1-2,1-3,2-3 \
  --allow-missing-depth \
  --checkpoint weights/pairs_ros1_sslnet_audio/best_model.pth \
  --indices all \
  --predictions-csv reports/pairs_ros1_predictions.csv \
  --distributions-npz reports/pairs_ros1_distributions.npz
```

测试输出包含 `Peak DOA MAE` 与 `Peak distance MAE`。

## 3. 不使用模型先验证 Audio Map

当还没有训练好的 SSLNet checkpoint 时，使用 fake 节点验证全局融合是否正确。fake 节点
可把 bag 最后一帧 odom 的位置视为固定声源，并根据每个当前 odom 生成带噪声的
DOA/distance 分布。

终端 1：

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

终端 2：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py \
  _odom_topic:=/lio/odom \
  _map_center_x:=0.0 \
  _map_center_y:=0.0 \
  _map_size_m:=12.0 \
  _resolution:=0.05 \
  _min_confidence:=0.0
```

终端 3：

```bash
rosparam set use_sim_time true
rosbag play --clock /path/to/test.bag
```

关键输出：

```text
/sslnet_fake_prediction/markers       绿色 simulated ground truth 声源
/sslnet_audio_map/markers              audio heatmap、红色 argmax、机器人及轨迹
/sslnet_audio_map/status               融合输入与时间匹配状态
```

## 4. 使用真实模型生成 Audio Map

首先启动 ReSpeaker 采集，且采集格式必须与训练一致，通常为 `16000 Hz`、6 通道：

```bash
source /opt/ros/noetic/setup.bash
source /home/kemove/yyz/respeaker_ros/devel/setup.bash
roslaunch respeaker_ros_recorder respeaker_collect.launch
```

启动实时音频推理：

```bash
cd /home/kemove/yyz/audio-nav/respeaker_ros

python deploy/ros1_sslnet_audio_node.py \
  _checkpoint:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.pth \
  _device:=cuda:0 \
  _window_seconds:=1.0 \
  _hop_seconds:=0.5
```

如需将网络前向改为 TensorRT，先在含 `onnx`、Python TensorRT 与 CUDA 的环境中导出：

```bash
python deploy/export_sslnet_tensorrt.py \
  --checkpoint weights/pairs_ros1_sslnet_audio/best_model.pth \
  --onnx weights/pairs_ros1_sslnet_audio/best_model.onnx \
  --engine weights/pairs_ros1_sslnet_audio/best_model.engine \
  --window-seconds 1.0 \
  --device cuda:0 \
  --builder python

python deploy/ros1_sslnet_audio_engine_node.py \
  _engine:=$(pwd)/weights/pairs_ros1_sslnet_audio/best_model.engine \
  _device:=cuda:0 \
  _window_seconds:=1.0 \
  _hop_seconds:=0.5
```

engine 节点发布与 `.pth` 节点相同的预测 topic，因此下面的 audio map 启动命令不变。
导出还会产生 `best_model.engine.json` 预处理 metadata；它必须随 engine 一同部署。

启动 audio map：

```bash
python deploy/ros1_sslnet_audio_map_fusion.py
```

该节点默认订阅 `/Odometry`，地图中心为 `(0.0, 0.0)`，地图大小为
`12.0 m x 12.0 m`，分辨率为 `0.05 m/cell`。

实时推理输出：

```text
/sslnet_audio_inference/prediction_json
/sslnet_audio_inference/doa_distribution
/sslnet_audio_inference/distance_distribution
```

全局融合输出：

```text
/sslnet_audio_map/map
/sslnet_audio_map/markers
/sslnet_audio_map/argmax
/sslnet_audio_map/status
```

详细的音频窗口、LiDAR 转换、marker 和参数说明见 [deploy/README.md](deploy/README.md)。

## 5. 运行 YOLOE Visual Map

visual map 订阅 RGB、depth、CameraInfo 与 odom，将视觉物体投影到和 audio map 相同的
全局网格。当前默认实现使用目标中央区域的 median depth，每个检测只累计一个物体中心
位置，降低错误深度向背景扩散的问题。默认只将 depth 不超过 `4.0 m` 的视觉物体写入
地图；该筛选不改变 audio map 的 `0-6 m` 声学距离分布。

没有 `/camera/color/camera_info` 时，可先仅为流程测试发布近似内参：

```bash
python deploy_yolo/ros1_fake_camera_info.py \
  _rgb_topic:=/camera/color/image_raw \
  _output_topic:=/camera/color/camera_info \
  _hfov_deg:=69.0
```

当前实时测试配置已写入 visual map 默认参数：

```text
model = deploy_yolo/yoloe-11s-seg.engine
rgb = /camera/color/image_raw
depth = /camera/depth/image_rect_raw
camera_info = /camera/color/camera_info
odom = /Odometry
map_center = (0.0, 0.0), map_size_m = 12.0, resolution = 0.05
conf = 0.50, max_depth_m = 4.0, device = 0
projection_pose_source = odom, broadcast_odom_tf = false
project_mask_footprint = false, marker_cell_size_m = 0.15
```

因此可直接启动：

```bash
python deploy_yolo/ros1_yoloe_visual_map.py
```

注意：

- `_depth_topic` 应优先使用已经注册到 RGB 的 aligned depth。
- `.engine` 的类别在导出时已经固定，运行时 `_classes` 不会改变类别。
- `_projection_pose_source:=odom` 适合先验证地图流程，但假设相机位于 odom 原点且朝前。
- 需要准确外参时，改用 `_projection_pose_source:=tf` 并提供相机到全局地图的 TF。

输出：

```text
/yoloe_visual_map/map                全局视觉占据图
/yoloe_visual_map/markers            视觉 heatmap、检测点、机器人及轨迹
/yoloe_visual_map/annotated_image    分割/检测标注图
/yoloe_visual_map/detections_json    全局目标位置
/yoloe_visual_map/status             输入与处理状态
```

完整说明见 [deploy_yolo/README.md](deploy_yolo/README.md)。

## 6. RViz 联合可视化

在 RViz 中将 `Fixed Frame` 设置为 odom 的全局 frame，例如 `camera_init`。添加：

```text
MarkerArray: /sslnet_audio_map/markers
MarkerArray: /yoloe_visual_map/markers
PointCloud2: /livox/points_rviz
```

需要查看原始 OccupancyGrid 时还可添加：

```text
Map: /sslnet_audio_map/map
Map: /yoloe_visual_map/map
```

显示语义：

```text
audio map 彩色 heatmap / 红点     声音融合概率与预测声源 argmax
visual map 蓝绿色方格 / 标注点    RGB-D 识别并投影的目标占据区域
蓝色或青色机器人 marker/轨迹      odom 位姿和运动轨迹
LiDAR 点云                         环境几何参考
```

若 `/livox/lidar` 是 `livox_ros_driver2/CustomMsg`，启动转换节点：

```bash
source /home/kemove/driver_ws/devel/setup.bash
python deploy/ros1_livox_custom_to_pointcloud2.py \
  _output_topic:=/livox/points_rviz \
  _z_max:=1.9
```

只有 TF 树确实缺少全局 frame 到 `livox_frame` 的变换，并确认 odom 表示 LiDAR 位姿时，
才在 audio 或 visual 节点之一设置：

```text
_broadcast_odom_tf:=true _sensor_frame_id:=livox_frame
```

不要在两个节点中同时广播相同 TF。

## 7. 快速排障

检查 audio map：

```bash
rostopic echo -n 1 /sslnet_audio_map/status
rostopic hz /sslnet_audio_map/map
```

检查 visual map：

```bash
rostopic echo -n 1 /yoloe_visual_map/status
rostopic hz /yoloe_visual_map/map
rostopic echo -n 1 /yoloe_visual_map/detections_json
```

常见判断：

| 现象 | 检查项 |
| --- | --- |
| `No module named 'respeaker_ros_recorder'` | 在运行终端 source ReSpeaker catkin 工作空间 |
| visual `confidence_threshold` 仍为 `0.8` | 删除 `/yoloe_visual_map/conf` ROS 私有参数；当前默认值为 `0.50` |
| 二维检测有值但 visual map 没点 | 查看 `latest_depth_rejected_count` 与 `latest_out_of_map_count` |
| `waiting_for_odom` | 确认 `_odom_topic` 与实际 topic 名一致 |
| `odom_map_frame_mismatch` | audio/visual map 与 odom 应使用同一个全局 frame |
| `TF_OLD_DATA` | bag 重播时使用 `--clock`，重启 RViz 与唯一 TF 广播节点，避免重复广播 |
| visual 历史位置点不明显 | 确认 `occupied_cell_count > 0`，调大 `_marker_cell_size_m` |

重置累计地图：

```bash
rosservice call /sslnet_audio_map/reset
rosservice call /yoloe_visual_map/reset
```
