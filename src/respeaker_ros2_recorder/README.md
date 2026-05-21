# respeaker_ros2_recorder

ROS 2 Python 3.10 package for publishing ReSpeaker multi-channel audio.

## Build

From a ROS 2 workspace:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select respeaker_ros2_recorder
source install/setup.bash
```

## Run

```bash
ros2 run respeaker_ros2_recorder respeaker_multichannel_node.py
```

With parameters:

```bash
ros2 run respeaker_ros2_recorder respeaker_multichannel_node.py --ros-args \
  -p sample_rate:=16000 \
  -p channels:=6 \
  -p chunk_size:=1600 \
  -p device_index:=-1 \
  -p frame_id:=respeaker \
  -p topic_name:=/respeaker/audio_raw
```

Launch the recorder:

```bash
ros2 launch respeaker_ros2_recorder respeaker_collect.launch.py
```

Launch and record a ROS 2 bag:

```bash
ros2 launch respeaker_ros2_recorder respeaker_collect.launch.py \
  record_bag:=true \
  bag_path:=respeaker_audio
```

List audio input devices:

```bash
ros2 run respeaker_ros2_recorder list_audio_devices.py
```

Export a ROS 2 bag topic to WAV:

```bash
ros2 run respeaker_ros2_recorder export_audio_from_bag.py \
  --bag /path/to/bag_dir \
  --topic /respeaker/audio_raw \
  --out respeaker_audio.wav
```

Batch export ROS 2 bags to WAV files. Output names include the first audio
timestamp by default, for example `bag_name_1712345678900000000.wav`:

```bash
ros2 run respeaker_ros2_recorder batch_export_audio_from_bags.py \
  --input-dir /media/kemove/T9/bag/clock \
  --output-dir wav_exports \
  --topic /respeaker/audio_raw
```

Use `--no-timestamp-in-name` to save files as `bag_name.wav`.
