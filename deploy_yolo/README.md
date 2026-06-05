# YOLOE RGB-D Visual Map ROS1 部署

总流程入口见仓库根目录 [`README.md`](../README.md)；SSLNet 实时音频与 audio map 详见
[`deploy/README.md`](../deploy/README.md)。

`ros1_yoloe_visual_map.py` 实时订阅 RGB 与对齐深度图，使用 YOLOE 实例分割识别指定
目标，在 mask 中央区域计算鲁棒 median depth，将每个物体的中心位置通过 odom 位姿或
TF 转换到全局坐标，持续累计一张 `visual map`。默认模式只标记物体出现的位置，不投影
整块 mask，这样更直接，也避免将边缘错误 depth 散射到其他位置。Visual map 是
视觉目标位置占据图：目标中心投影命中的网格写入固定占据值，而不是
像 audio map 一样表达高斯或声源概率分布。它默认独立维护与 Audio Map 相同规格的地图：
`12.0 m x 12.0 m`、`0.05 m/cell`。需要严格逐格叠加时，为两个节点设置相同的固定
地图中心；也可以显式启用订阅 `/sslnet_audio_map/map` 来复制其几何。
节点默认只累计 median depth 在 `4.0 m` 内的视觉物体；更远的二维检测仍可出现在
annotated image 中，但不会写入 visual map。
图像订阅回调只保留最新同步 RGB-D 帧，后台线程执行 YOLOE；推理速度低于相机帧率时会
主动丢弃旧帧，避免实时地图持续融合过期图像。
在线节点对常见 raw `sensor_msgs/Image` 编码使用 NumPy 直接解码，不依赖
`cv_bridge`，以避免 conda 环境与 ROS OpenCV/GDAL/TIFF 动态库冲突。

本目录还提供 `ros1_fake_camera_info.py`，可在相机没有发布 `CameraInfo` 时用估算内参
先跑通 RGB-D 投影和 RViz 流程。它不能代替真实相机标定。

## 输入要求

默认输入 topic：

```text
/camera/color/image_raw       sensor_msgs/Image，RGB 图像
/camera/depth/image_rect_raw  sensor_msgs/Image，与 RGB 像素对齐的深度图
/camera/color/camera_info     sensor_msgs/CameraInfo，RGB/对齐深度相机内参
/Odometry                     nav_msgs/Odometry，未收到 audio map 时用于初始化地图
/sslnet_audio_map/map         nav_msgs/OccupancyGrid，默认用于复制 audio map 的 frame/origin/resolution
```

深度必须已注册到 RGB 图像：同一个像素位置应代表同一个物体点。如果相机驱动提供
`aligned_depth_to_color` 一类 topic，请优先把 `_depth_topic` 指向对齐后的深度话题。
当 RGB 与 depth 图像尺寸不同，节点默认保留 depth 的原始测距值和分辨率，将 RGB 缩放
至 depth 尺寸后执行分割（`_resize_rgb_to_depth:=true`），并同步缩放 RGB 内参后反投影。
这样不会通过放大 depth 复制或插值深度边界；尺寸相同仍不代表两个相机视场已经完成
空间配准，精确地图仍应使用注册到 RGB 的 depth。
投影所使用的坐标 frame 取自 `_camera_frame` 参数或 `CameraInfo.header.frame_id`；
fake CameraInfo 会沿用 RGB 图像的 frame。

默认 `_projection_pose_source:=odom`，节点直接使用 `/Odometry` 位姿投影目标，不要求
相机 TF。此模式假设相机位于 odom 描述的机器人原点且朝机器人正前方安装，并按 optical
frame 轴定义转换为平面坐标：相机深度 `z` 对应机器人前方 `x`，图像右侧 `x` 对应
机器人右侧 `-y`。它适合先验证 visual map 累积流程；相机存在安装偏移或旋转时，地图
位置会产生误差。

如果 RViz 同时显示位于 `livox_frame` 的点云并提示
`No transform from [camera_init] to [livox_frame]`，且 `/Odometry` pose 确实表示
LiDAR 在 `camera_init` 中的位姿，可以仅在一个 map 节点上启用：

```bash
_broadcast_odom_tf:=true _sensor_frame_id:=livox_frame
```

audio 与 visual 节点同时运行时不要在两边同时开启该选项，也不要与已有 LIO TF 重复广播。
如果重播 bag、从较早时间重新播放，保留着上一轮 TF 缓存的 RViz 会报告
`TF_OLD_DATA`。用 `rosbag play --clock` 配合仿真时间，并在重新开始回放时重启
RViz 和负责广播该 TF 的唯一节点；仅验证 visual map 发布时可显式关闭
`_broadcast_odom_tf:=false`。

需要准确相机外参时，设置 `_projection_pose_source:=tf`。此时 TF 树必须能够将
RGB/depth 的相机 frame 转换到 `/sslnet_audio_map/map/header.frame_id` 或独立地图
frame，例如从相机 optical frame 转换到 `camera_init` 或 `odom`。

## 没有 CameraInfo 时临时测试

样本 RGB 图像为 `640 x 480`。如果实时 RGB topic 也为该分辨率、但没有
`/camera/color/camera_info`，先开一个终端运行 fake 内参节点：

```bash
cd /home/kemove/yyz/audio-nav/respeaker_ros
source /opt/ros/noetic/setup.bash

python deploy_yolo/ros1_fake_camera_info.py \
  _rgb_topic:=/camera/color/image_raw \
  _output_topic:=/camera/color/camera_info \
  _hfov_deg:=69.0
```

它订阅 RGB 图像以自动读取真实宽高和 `header.frame_id`，默认按 pinhole 模型估算：

```text
fx = width / (2 * tan(hfov / 2))
fy = fx
cx = (width - 1) / 2
cy = (height - 1) / 2
distortion = 0
```

如果已知更接近的内参，可以覆盖估算值：

```bash
python deploy_yolo/ros1_fake_camera_info.py \
  _rgb_topic:=/camera/color/image_raw \
  _output_topic:=/camera/color/camera_info \
  _fx:=615.0 \
  _fy:=615.0 \
  _cx:=319.5 \
  _cy:=239.5
```

确认 fake topic 已有输出：

```bash
rostopic echo -n 1 /camera/color/camera_info
```

注意：假内参足够用于确认 YOLOE、depth、TF 和 visual map 的程序链路能运行，但焦距和
畸变不真实会导致物体全局位置产生横向偏差。判断 visual/audio map 是否精准对齐前，
应替换为相机的真实标定 `CameraInfo`。

## YOLOE 环境

YOLOE segmentation 由 Ultralytics 提供。使用支持 YOLOE 的较新版本：

```bash
python -m pip install -U ultralytics
```

官方 YOLOE segmentation 权重示例：

```text
yoloe-11s-seg.pt
yoloe-11m-seg.pt
yoloe-v8s-seg.pt
```

`.pt` 模型使用 text prompts 指定目标类别，默认仅寻找 `person`；可通过 `_classes`
修改。节点也支持 TensorRT `.engine`。运行时 `_classes` 还会作为后过滤器使用，
只有匹配类别会写入 visual map，并出现在 `/yoloe_visual_map/annotated_image` 中：

- segmentation engine 使用 mask 中央区域的 median depth 定位物体中心，减少边缘背景深度污染。
- detection engine 没有 mask 时，节点使用检测框内部区域的有效深度估计物体位置。
- `.engine` 的类别在导出时已经固定，运行时 `_classes` 不能新增 engine 未包含的类别，
  但可以从 engine 输出类别中筛选，例如只保留 `guitar`。

导出 YOLOE TensorRT segmentation engine 时，应在 `.pt` 模型上先设置目标类别：

```python
from ultralytics import YOLOE

model = YOLOE("yoloe-11s-seg.pt")
model.set_classes(["person", "chair", "bottle"])
model.export(format="engine", imgsz=640)
```

## 运行

当前验证通过的实时配置已写入两个 map 的默认参数：默认 odom 为 `/Odometry`，固定地图
中心为 `(0.0, 0.0)`，地图大小为 `12.0 m x 12.0 m`，分辨率为 `0.05 m/cell`。
audio map 可直接启动：

```bash
cd /home/kemove/yyz/audio-nav/respeaker_ros
source /opt/ros/noetic/setup.bash

python deploy/ros1_sslnet_audio_map_fusion.py
```

另一个终端直接启动 visual map；它默认加载本目录的
`yoloe-11s-seg.engine`，使用 `/camera/depth/image_rect_raw`、GPU `0`、
`conf=0.50` 和 `4.0 m` 最大入图深度，并且不需要订阅 audio map：

```bash
cd /home/kemove/yyz/audio-nav/respeaker_ros
source /opt/ros/noetic/setup.bash

python deploy_yolo/ros1_yoloe_visual_map.py
```

如需改用 bag 中的 `/lio/odom`，应在 audio 与 visual 两个命令中均传入
`_odom_topic:=/lio/odom`。

如果希望 visual map 直接复制正在运行的 audio map 几何，可改为：

```bash
python deploy_yolo/ros1_yoloe_visual_map.py \
  _model:=yoloe-11s-seg.pt \
  _classes:=person,chair,bottle \
  _use_audio_map_geometry:=true \
  _audio_map_topic:=/sslnet_audio_map/map \
  _odom_topic:=/Odometry
```

使用已导出的 TensorRT segmentation 或 detection engine：

```bash
python deploy_yolo/ros1_yoloe_visual_map.py \
  _model:=/path/to/another-yoloe-seg.engine
```

如果 depth 为毫米单位的 `16UC1`，节点会自动使用 `0.001` 的比例换算为米；若你的深度
编码不同但数值仍为毫米，可显式指定：

```bash
_depth_scale:=0.001
```

## 发布 Topic

```text
/yoloe_visual_map/map
  nav_msgs/OccupancyGrid，全局视觉目标占据图，几何与 audio map 一致

/yoloe_visual_map/markers
  visualization_msgs/MarkerArray，蓝绿色 occupied cells、目标位置与标签、机器人位姿和轨迹

/yoloe_visual_map/annotated_image
  sensor_msgs/Image，YOLOE mask/box 标注后的 RGB 图

/yoloe_visual_map/detections_json
  String，每帧目标类别、置信度和全局 x/y/z 位置

/yoloe_visual_map/status
  String，模型、输入与处理状态
```

## RViz 叠加显示

沿用当前已经能正确显示 audio marker 与 LiDAR 的 `Fixed Frame`，添加：

```text
MarkerArray: /sslnet_audio_map/markers
MarkerArray: /yoloe_visual_map/markers
PointCloud2: /livox/points_rviz
```

如果希望同时观察 OccupancyGrid 数据，也可添加：

```text
Map: /sslnet_audio_map/map
Map: /yoloe_visual_map/map
```

彩色 marker 更适合直接比较二者位置，因为两类 marker 都在各自 map 的全局坐标中发布。
在 visual marker 中，青色箭头为最新 `/Odometry` 位姿，青色线为累计机器人轨迹。
Visual heatmap 使用蓝绿色方块显示累计目标中心位置；底层网格仍为 `0.05 m`，marker
默认显示为 `0.15 m` 大小以便在 RViz 中看清。

## 排查

查看节点是否已读取图像、内参和地图几何：

```bash
rostopic echo -n 1 /yoloe_visual_map/status
```

状态中的 `frames_inferred` 表示模型已经对图像完成预测，
`latest_raw_detection_count` 表示当前图像上的二维检测数量，`frames_processed` 表示
预测结果已经完成坐标投影并写入 visual map。地图几何初始化后，节点会先发布空的
`/yoloe_visual_map/map` 与 `/yoloe_visual_map/markers`；随后只有能够完成投影的帧
才会更新占据区域。`/yoloe_visual_map/annotated_image` 在二维推理完成后即可用于检查
识别效果。`occupied_cell_count` 表示累计视觉占据网格数量；物体区域投影成功时它应
逐步增加。
`confidence_threshold` 与 `broadcast_odom_tf` 表示本次实际生效的私有参数。ROS 参数
服务器可能保留之前通过 `_conf:=...` 或 `rosparam set` 设置的值；脚本中的默认值只在
该参数不存在时使用。如果启动日志仍显示 `conf=0.800`，可清理后显式指定：

```bash
rosparam delete /yoloe_visual_map/conf 2>/dev/null || true
rosparam delete /yoloe_visual_map/broadcast_odom_tf 2>/dev/null || true

python deploy_yolo/ros1_yoloe_visual_map.py \
  _model:=/path/to/yoloe-11s-seg.engine \
  _conf:=0.50 \
  _projection_pose_source:=odom \
  _broadcast_odom_tf:=false
```

当 `_projection_pose_source:=odom` 时，视觉物体写入地图不依赖 TF；若此时
`frames_processed` 持续增加但 `latest_detection_count` 为零，优先检查阈值、分割
输出与有效深度，而不是 TF。以下计数可以直接解释检测未写入地图的原因：

```text
latest_raw_detection_count    二维 YOLOE 检测数
latest_depth_rejected_count   找不到有效 depth，无法得到全局位置的检测数
latest_detection_count        已得到全局位置的检测数
latest_out_of_map_count       全局位置在当前地图范围外的检测数
latest_map_updated_count      当前帧实际写入地图的物体数
```

实时相机默认使用 `_image_pairing_mode:=latest`：每个 RGB 帧配对最近收到的 depth，
避免两路 timestamp 不完全一致时 `ApproximateTimeSynchronizer` 只触发少数帧。
`rgb_messages_received`、`depth_messages_received` 应持续增长，`frames_received` 表示
进入推理队列的 RGB-D 对数量，`last_rgb_depth_dt_sec` 表示最近一次配对的时间差。
如果 `frames_received` 持续增加而 `frames_inferred` 长时间不增长，应检查模型加载、
TensorRT 环境或 GPU 推理是否阻塞。TF 查询默认使用 `_tf_timeout_sec:=0.0` 非阻塞
检查，仅在 `_projection_pose_source:=tf` 时使用；缺少 TF 时只跳过该帧的地图融合，
不会停止后续图像接收。

常见状态：

```text
waiting_for_camera_info       未收到相机内参
waiting_for_audio_map_geometry 默认坐标模式下尚未收到 /sslnet_audio_map/map
waiting_for_odom              未收到 odom，无法确定独立地图的 frame
waiting_for_depth_image       latest 配对模式下尚未收到 depth 图像
waiting_for_recent_depth_image 最近 depth 与 RGB 时间差超过允许值
odom_map_frame_mismatch       odom frame 与 visual map frame 不一致
waiting_for_camera_to_map_tf  缺少相机 frame 到全局 map frame 的 TF
rgb_depth_not_aligned         RGB/depth 尺寸不同且关闭了 `_resize_rgb_to_depth`
image_decode_failed           输入不是支持的 raw 图像编码，或消息数据长度异常
model_load_failed             Ultralytics 版本或 YOLOE 权重不可用
inference_failed              YOLOE 执行失败
processing_failed             深度投影或地图更新发生异常
running                       正常处理 RGB-D 帧
```

重置累计的 visual map：

```bash
rosservice call /yoloe_visual_map/reset
```

常用参数：

```text
_conf:=0.10                 YOLOE 检测筛选阈值；也兼容 _confidence_threshold/_conf_thres/_conf_thred
_imgsz:=640                 推理输入尺寸
_device:=0                  GPU；CPU 可设置为 cpu
_model:=deploy_yolo/yoloe-11s-seg.engine 默认使用本目录导出的 TensorRT segmentation engine
_classes:=guitar           运行时后过滤类别；只把该类别写入 map/annotated image
_depth_topic:=/camera/depth/image_rect_raw 默认 depth 输入 topic
_depth_camera_info_topic:=/camera/depth/camera_info 默认 depth 内参；RGB resize 到 depth 时用于反投影
_odom_topic:=/Odometry      默认 odom 输入 topic
_min_depth_m:=0.20          使用的最小深度
_max_depth_m:=4.0           每帧仅将 4 m 以内的检测目标写入 visual map
_resize_rgb_to_depth:=true  尺寸不同时，缩放 RGB 至 depth 尺寸并保留原始深度
_image_pairing_mode:=latest 实时输入按最近 depth 配对；严格同步可设置 approximate
_max_rgb_depth_age_sec:=0.25 latest 模式允许的最大 RGB-depth 时间差；负值不检查
_sync_slop_sec:=0.05        approximate 模式下的时间戳容差
_output_stamp_mode:=current RViz 输出时间戳；current 可避免旧图像时间导致 MessageFilter 丢弃
_projection_pose_source:=odom 用 odom 直接定位；准确外参投影可设置 tf
_max_odom_diff_sec:=0.50    odom 投影时 RGB 帧可匹配的最大 odom 时间差
_camera_offset_x/y/z:=0     odom 投影模式下，相机坐标相对 odom child/body 的平移外参，单位 m
_camera_roll_deg/_camera_pitch_deg/_camera_yaw_deg:=0 odom 投影模式下，相机相对 odom child/body 的旋转外参
_tf_timeout_sec:=0.0        TF 非阻塞查询；bag/仿真时间下避免缺 TF 阻塞 worker
_broadcast_odom_tf:=false   依据 odom 广播全局 frame 到传感器 frame 的 TF
_sensor_frame_id:=          广播 TF 时的 child frame，例如 livox_frame
_center_depth_region_scale:=0.30 使用 mask 中央区域占边界框的比例计算 median depth
_min_mask_depth_points:=8  用于计算 median depth 的最少有效深度点数
_project_mask_footprint:=false 默认仅投影中心点；true 时投影 median-depth mask 区域
_footprint_stride_px:=6    mask 投影的像素采样步长
_max_footprint_points:=1200 单个目标单帧最多投影的 mask 采样点数
_footprint_inflation_m:=0.0 中心点写入后的累计膨胀半径；默认不膨胀
_occupied_value:=1.0       物体中心投影命中格子的固定占据值
_marker_threshold:=0.10     RViz 中显示占据格子的最低值
_marker_cell_size_m:=0.15  RViz 中累计位置方块显示大小，不改变地图分辨率
_robot_marker_height:=0.10 robot 箭头在地图平面上的显示高度
_robot_path_length:=1000   visual marker 中保留的 odom 轨迹点数
_map_size_m:=12.0           独立地图边长，默认与 audio map 一致
_resolution:=0.05           独立地图分辨率，默认与 audio map 一致
_map_center_x/_map_center_y 固定全局地图中心，默认均为 0.0
_use_audio_map_geometry:=true 是否订阅并复制 audio map 几何；默认 true，保证 visual/audio map 坐标一致
```

支持直接解码的 raw 图像编码：

```text
RGB:   rgb8, bgr8, rgba8, bgra8, mono8, 8UC1
Depth: mono16, 16UC1, 16SC1, 32FC1
```

参考：Ultralytics YOLOE 官方说明文档：
https://docs.ultralytics.com/models/yoloe/
