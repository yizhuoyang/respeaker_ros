#!/usr/bin/env python3
"""Run YOLOE instance segmentation on RGB-D frames and maintain a global visual map."""

import json
import math
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
import threading

import cv2
import numpy as np


def image_encoding_info(encoding):
    """Return numpy dtype and channel count for common raw ROS image encodings."""
    encoding = str(encoding).lower()
    if encoding in ("rgb8", "bgr8"):
        return np.uint8, 3
    if encoding in ("rgba8", "bgra8"):
        return np.uint8, 4
    if encoding in ("mono8", "8uc1"):
        return np.uint8, 1
    if encoding in ("mono16", "16uc1", "16sc1"):
        return np.uint16 if encoding != "16sc1" else np.int16, 1
    if encoding == "32fc1":
        return np.float32, 1
    raise RuntimeError("Unsupported raw image encoding: %s" % encoding)


def ros_image_to_numpy(message):
    """Decode a raw sensor_msgs/Image without loading cv_bridge/OpenCV ROS libraries."""
    dtype, channels = image_encoding_info(message.encoding)
    itemsize = np.dtype(dtype).itemsize
    row_bytes = int(message.width) * channels * itemsize
    if int(message.step) < row_bytes:
        raise RuntimeError(
            "Image step=%s is smaller than expected row size=%s for encoding=%s"
            % (message.step, row_bytes, message.encoding)
        )
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required_bytes = int(message.height) * int(message.step)
    if raw.size < required_bytes:
        raise RuntimeError(
            "Image data size=%s is smaller than height*step=%s" % (raw.size, required_bytes)
        )
    rows = raw[:required_bytes].reshape(int(message.height), int(message.step))
    packed = rows[:, :row_bytes].copy()
    image = packed.reshape(int(message.height), int(message.width), channels, itemsize)
    image = image.view(dtype).reshape(int(message.height), int(message.width), channels)
    if bool(message.is_bigendian) != (np.dtype(dtype).byteorder == ">"):
        if itemsize > 1 and bool(message.is_bigendian):
            image = image.byteswap()
    if channels == 1:
        image = image[:, :, 0]
    return image


def ros_rgb_to_bgr(message):
    """Decode a raw ROS color image into the BGR array expected by Ultralytics."""
    image = ros_image_to_numpy(message)
    encoding = str(message.encoding).lower()
    if encoding == "rgb8":
        return image[:, :, ::-1].copy()
    if encoding == "rgba8":
        return image[:, :, [2, 1, 0]].copy()
    if encoding == "bgra8":
        return image[:, :, :3].copy()
    if encoding == "bgr8":
        return image
    if encoding in ("mono8", "8uc1"):
        return np.repeat(image[:, :, None], 3, axis=2)
    raise RuntimeError("RGB topic has unsupported color encoding: %s" % message.encoding)


def quaternion_to_rotation_matrix(quaternion):
    """Convert a ROS quaternion to a 3-by-3 rotation matrix."""
    x = float(quaternion.x)
    y = float(quaternion.y)
    z = float(quaternion.z)
    w = float(quaternion.w)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-12:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def transform_points(points, transform):
    """Transform Nx3 camera-frame points using geometry_msgs/TransformStamped."""
    rotation = quaternion_to_rotation_matrix(transform.transform.rotation)
    translation = np.asarray(
        [
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ],
        dtype=np.float32,
    )
    return points.dot(rotation.T) + translation


def optical_points_to_robot(points):
    """Convert forward-facing optical-frame points into robot-frame axes."""
    return np.column_stack((points[:, 2], -points[:, 0], -points[:, 1])).astype(
        np.float32
    )


def transform_points_with_pose(points, pose):
    """Transform Nx3 points with an odometry pose expressed in the map frame."""
    rotation = quaternion_to_rotation_matrix(pose.orientation)
    translation = np.asarray(
        [pose.position.x, pose.position.y, pose.position.z],
        dtype=np.float32,
    )
    return points.dot(rotation.T) + translation


def color_for_class(class_id):
    """Return a stable bright RGB color for a target class."""
    colors = (
        (0.15, 0.95, 0.30),
        (1.00, 0.72, 0.08),
        (0.10, 0.75, 1.00),
        (1.00, 0.25, 0.75),
        (0.75, 0.35, 1.00),
        (0.98, 0.42, 0.08),
    )
    return colors[int(class_id) % len(colors)]


def normalize_class_name(name):
    return str(name).strip().lower().replace("_", " ")


def parse_class_filter(class_text):
    tokens = [text.strip() for text in str(class_text).split(",") if text.strip()]
    if not tokens or any(token.lower() in ("*", "all", "none") for token in tokens):
        return tokens, set()
    return tokens, {normalize_class_name(token) for token in tokens}


class VisualMapGeometry:
    def __init__(self, frame_id, resolution, width, height, origin_x, origin_y):
        self.frame_id = str(frame_id)
        self.resolution = float(resolution)
        self.width = int(width)
        self.height = int(height)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)

    def equivalent(self, other):
        return (
            other is not None
            and self.frame_id == other.frame_id
            and self.width == other.width
            and self.height == other.height
            and abs(self.resolution - other.resolution) < 1e-9
            and abs(self.origin_x - other.origin_x) < 1e-6
            and abs(self.origin_y - other.origin_y) < 1e-6
        )

    def points_to_cells(self, xy_points):
        columns = np.floor((xy_points[:, 0] - self.origin_x) / self.resolution).astype(np.int32)
        rows = np.floor((xy_points[:, 1] - self.origin_y) / self.resolution).astype(np.int32)
        valid = (
            (columns >= 0)
            & (columns < self.width)
            & (rows >= 0)
            & (rows < self.height)
        )
        return rows[valid], columns[valid]

    def cell_center(self, row, column):
        return (
            self.origin_x + (float(column) + 0.5) * self.resolution,
            self.origin_y + (float(row) + 0.5) * self.resolution,
        )


class YOLOEVisualMapNode:
    def __init__(self):
        import message_filters
        import rospy
        import tf2_ros
        from nav_msgs.msg import OccupancyGrid, Odometry
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String
        from std_srvs.srv import Empty
        from visualization_msgs.msg import MarkerArray

        self.rospy = rospy
        self.lock = threading.Lock()
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(20.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.rgb_topic = rospy.get_param("~rgb_topic", "/camera/color/image_raw")
        self.depth_topic = rospy.get_param("~depth_topic", "/camera/depth/image_rect_raw")
        self.camera_info_topic = rospy.get_param(
            "~camera_info_topic", "/camera/color/camera_info"
        )
        self.odom_topic = rospy.get_param("~odom_topic", "/Odometry")
        self.use_audio_map_geometry = bool(rospy.get_param("~use_audio_map_geometry", False))
        self.audio_map_topic = rospy.get_param("~audio_map_topic", "/sslnet_audio_map/map")
        self.camera_frame_override = rospy.get_param("~camera_frame", "")
        self.map_frame_override = rospy.get_param("~map_frame", "")
        self.projection_pose_source = str(
            rospy.get_param("~projection_pose_source", "odom")
        ).strip().lower()
        if self.projection_pose_source not in ("odom", "tf"):
            raise ValueError("~projection_pose_source must be 'odom' or 'tf'.")

        default_model_path = str(Path(__file__).resolve().parent / "yoloe-11s-seg.engine")
        self.model_path = rospy.get_param("~model", default_model_path)
        self.exported_engine = Path(self.model_path).suffix.lower() == ".engine"
        class_text = rospy.get_param("~classes", "person")
        self.classes, self.class_filter = parse_class_filter(class_text)
        self.class_filter_enabled = bool(self.class_filter)
        self.confidence_threshold = float(rospy.get_param("~conf", 0.5))
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("~conf must be between 0.0 and 1.0.")
        self.image_size = int(rospy.get_param("~imgsz", 640))
        self.device = rospy.get_param("~device", "0")

        self.depth_scale = float(rospy.get_param("~depth_scale", 0.0))
        self.resize_rgb_to_depth = bool(rospy.get_param("~resize_rgb_to_depth", True))
        self.min_depth_m = float(rospy.get_param("~min_depth_m", 0.20))
        self.max_depth_m = float(rospy.get_param("~max_depth_m", 4.0))
        self.min_mask_depth_points = max(
            int(rospy.get_param("~min_mask_depth_points", 8)), 1
        )
        self.center_depth_region_scale = float(
            np.clip(rospy.get_param("~center_depth_region_scale", 0.30), 0.05, 1.0)
        )
        self.project_mask_footprint = bool(rospy.get_param("~project_mask_footprint", False))
        self.footprint_stride_px = max(
            int(rospy.get_param("~footprint_stride_px", 6)), 1
        )
        self.max_footprint_points = max(
            int(rospy.get_param("~max_footprint_points", 1200)), 1
        )
        self.footprint_inflation_m = max(
            float(rospy.get_param("~footprint_inflation_m", 0.0)), 0.0
        )
        self.tf_timeout = max(float(rospy.get_param("~tf_timeout_sec", 0.0)), 0.0)

        self.map_size_m = float(rospy.get_param("~map_size_m", 12.0))
        self.resolution = float(rospy.get_param("~resolution", 0.05))
        map_center_x = rospy.get_param("~map_center_x", 0.0)
        map_center_y = rospy.get_param("~map_center_y", 0.0)
        if (map_center_x is None) != (map_center_y is None):
            raise ValueError("Set both ~map_center_x and ~map_center_y, or neither.")
        self.map_center = (
            (float(map_center_x), float(map_center_y))
            if map_center_x is not None
            else None
        )
        self.occupied_value = float(
            np.clip(rospy.get_param("~occupied_value", 1.0), 0.0, 1.0)
        )
        self.publish_marker_threshold = float(
            np.clip(rospy.get_param("~marker_threshold", 0.10), 0.0, 1.0)
        )
        self.marker_max_cells = max(int(rospy.get_param("~marker_max_cells", 6000)), 1)
        self.marker_height = float(rospy.get_param("~marker_height", 0.04))
        self.marker_cell_size_m = max(
            float(rospy.get_param("~marker_cell_size_m", 0.15)), self.resolution
        )
        self.object_marker_height = float(rospy.get_param("~object_marker_height", 0.18))
        self.robot_marker_height = float(rospy.get_param("~robot_marker_height", 0.10))
        self.robot_path = deque(
            maxlen=max(int(rospy.get_param("~robot_path_length", 1000)), 1)
        )
        self.broadcast_odom_tf = bool(rospy.get_param("~broadcast_odom_tf", False))
        self.sensor_frame_id = rospy.get_param("~sensor_frame_id", "")
        self.tf_broadcaster = (
            tf2_ros.TransformBroadcaster() if self.broadcast_odom_tf else None
        )
        self.image_pairing_mode = str(
            rospy.get_param("~image_pairing_mode", "latest")
        ).strip().lower()
        if self.image_pairing_mode not in ("latest", "approximate"):
            raise ValueError("~image_pairing_mode must be 'latest' or 'approximate'.")
        self.max_rgb_depth_age_sec = float(
            rospy.get_param("~max_rgb_depth_age_sec", 0.25)
        )

        self.geometry = None
        self.geometry_from_audio_map = False
        self.visual_map = None
        self.latest_odom = None
        self.latest_camera_info = None
        self.model = None
        self.model_error = ""
        self.last_status = "initializing"
        self.rgb_messages_received = 0
        self.depth_messages_received = 0
        self.frames_received = 0
        self.frames_inferred = 0
        self.frames_processed = 0
        self.latest_raw_detection_count = 0
        self.latest_detection_count = 0
        self.latest_depth_rejected_count = 0
        self.latest_out_of_map_count = 0
        self.latest_map_updated_count = 0
        self.last_object_marker_count = 0
        self.latest_depth_msg = None
        self.last_rgb_depth_dt_sec = None
        self.shutdown_timeout = float(rospy.get_param("~shutdown_timeout", 0.5))
        self.frame_queue = Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(
            target=self.inference_loop,
            name="yoloe_visual_map_worker",
            daemon=True,
        )

        self.map_pub = rospy.Publisher("~map", OccupancyGrid, queue_size=1, latch=True)
        self.marker_pub = rospy.Publisher("~markers", MarkerArray, queue_size=1, latch=True)
        self.annotated_pub = rospy.Publisher("~annotated_image", Image, queue_size=1)
        self.detections_pub = rospy.Publisher("~detections_json", String, queue_size=2)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)

        self.audio_map_sub = None
        if self.use_audio_map_geometry:
            self.audio_map_sub = rospy.Subscriber(
                self.audio_map_topic, OccupancyGrid, self.audio_map_callback, queue_size=1
            )
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)
        self.camera_info_sub = rospy.Subscriber(
            self.camera_info_topic, CameraInfo, self.camera_info_callback, queue_size=1
        )
        self.image_sync = None
        if self.image_pairing_mode == "approximate":
            self.rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
            self.depth_sub = message_filters.Subscriber(self.depth_topic, Image)
            self.rgb_sub.registerCallback(self.rgb_input_callback)
            self.depth_sub.registerCallback(self.depth_input_callback)
            sync_slop = float(rospy.get_param("~sync_slop_sec", 0.05))
            sync_queue = max(int(rospy.get_param("~sync_queue_size", 5)), 1)
            self.image_sync = message_filters.ApproximateTimeSynchronizer(
                [self.rgb_sub, self.depth_sub], queue_size=sync_queue, slop=sync_slop
            )
            self.image_sync.registerCallback(self.rgb_depth_callback)
        else:
            self.rgb_sub = rospy.Subscriber(
                self.rgb_topic, Image, self.rgb_latest_callback, queue_size=1
            )
            self.depth_sub = rospy.Subscriber(
                self.depth_topic, Image, self.depth_input_callback, queue_size=1
            )
        self.reset_service = rospy.Service("~reset", Empty, self.reset_callback)
        self.status_timer = rospy.Timer(rospy.Duration(1.0), self.status_callback)
        rospy.on_shutdown(self.shutdown)
        self.worker.start()

        rospy.loginfo(
            "YOLOE visual map started: rgb=%s depth=%s camera_info=%s use_audio_map_geometry=%s",
            self.rgb_topic,
            self.depth_topic,
            self.camera_info_topic,
            self.use_audio_map_geometry,
        )
        rospy.loginfo(
            "Visual map outputs: map=%s markers=%s annotated=%s status=%s",
            rospy.resolve_name("~map"),
            rospy.resolve_name("~markers"),
            rospy.resolve_name("~annotated_image"),
            rospy.resolve_name("~status"),
        )
        rospy.loginfo(
            "YOLOE inference processes only the newest RGB-D frame: pairing_mode=%s "
            "max_rgb_depth_age_sec=%.3f projection_pose_source=%s",
            self.image_pairing_mode,
            self.max_rgb_depth_age_sec,
            self.projection_pose_source,
        )
        rospy.loginfo(
            "YOLOE effective settings: conf=%.3f param=%s broadcast_odom_tf=%s "
            "sensor_frame_id=%s project_mask_footprint=%s footprint_inflation_m=%.3f",
            self.confidence_threshold,
            rospy.resolve_name("~conf"),
            self.broadcast_odom_tf,
            self.sensor_frame_id or "<odom child/default>",
            self.project_mask_footprint,
            self.footprint_inflation_m,
        )
        if self.projection_pose_source == "odom":
            rospy.logwarn(
                "Visual projection uses odom directly and assumes a forward-facing camera "
                "at the odometry pose origin; use ~projection_pose_source:=tf for calibrated extrinsics."
            )
        if self.tf_broadcaster is not None:
            rospy.logwarn(
                "Broadcasting odom pose as TF to sensor frame '%s'. Enable this on only one "
                "map node and only when odom describes that sensor pose.",
                self.sensor_frame_id or "odom.child_frame_id/livox_frame",
            )

    def load_model(self):
        if self.model is not None or self.model_error:
            return self.model is not None
        try:
            if self.exported_engine:
                from ultralytics import YOLO

                model_class = YOLO
            else:
                try:
                    from ultralytics import YOLOE

                    model_class = YOLOE
                except ImportError:
                    from ultralytics import YOLO

                    model_class = YOLO
            self.model = model_class(self.model_path)
            if self.classes and not self.exported_engine:
                if not hasattr(self.model, "set_classes"):
                    raise RuntimeError(
                        "This Ultralytics model cannot accept YOLOE text prompts via set_classes()."
                    )
                self.model.set_classes(self.classes)
            elif self.classes and self.exported_engine:
                self.rospy.logwarn(
                    "TensorRT engine %s has static classes; ~classes=%s will be used "
                    "as a post-filter for annotated image, map accumulation, and markers.",
                    self.model_path,
                    self.classes,
                )
            self.rospy.loginfo(
                "Loaded YOLOE model=%s classes=%s class_filter=%s exported_engine=%s",
                self.model_path,
                "embedded in engine" if self.exported_engine else (self.classes or "default LVIS/prompts"),
                sorted(self.class_filter) if self.class_filter_enabled else "disabled",
                self.exported_engine,
            )
            return True
        except Exception as exc:
            self.model_error = str(exc)
            self.last_status = "model_load_failed"
            self.rospy.logerr(
                "Cannot load YOLOE model '%s': %s. Install a recent Ultralytics release "
                "with YOLOE support and provide a YOLOE segmentation weight such as "
                "yoloe-11s-seg.pt.",
                self.model_path,
                exc,
            )
            return False

    def camera_info_callback(self, msg):
        with self.lock:
            self.latest_camera_info = msg

    def odom_callback(self, msg):
        with self.lock:
            self.latest_odom = msg
            self.robot_path.append(
                (
                    float(msg.pose.pose.position.x),
                    float(msg.pose.pose.position.y),
                    msg.header.frame_id,
                )
            )
        if self.tf_broadcaster is not None:
            self.publish_odom_tf(msg)

    def publish_odom_tf(self, odom):
        from geometry_msgs.msg import TransformStamped

        parent_frame = odom.header.frame_id or self.map_frame_override or "odom"
        child_frame = self.sensor_frame_id or odom.child_frame_id or "livox_frame"
        if parent_frame == child_frame:
            return
        transform = TransformStamped()
        transform.header.stamp = odom.header.stamp
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = odom.pose.pose.position.z
        transform.transform.rotation = odom.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def audio_map_callback(self, msg):
        geometry = VisualMapGeometry(
            msg.header.frame_id or self.map_frame_override or "odom",
            msg.info.resolution,
            msg.info.width,
            msg.info.height,
            msg.info.origin.position.x,
            msg.info.origin.position.y,
        )
        initialized = False
        with self.lock:
            changed = not geometry.equivalent(self.geometry)
            self.geometry = geometry
            self.geometry_from_audio_map = True
            if changed or self.visual_map is None:
                self.visual_map = np.zeros((geometry.height, geometry.width), dtype=np.float32)
                initialized = True
                self.rospy.loginfo(
                    "Visual map adopted audio map geometry: frame=%s size=%dx%d resolution=%.3f",
                    geometry.frame_id,
                    geometry.width,
                    geometry.height,
                    geometry.resolution,
                )
        if initialized:
            self.publish_map_snapshot(geometry, msg.header.stamp)

    def initialize_fallback_geometry(self):
        with self.lock:
            if self.geometry is not None:
                return True
            odom = self.latest_odom
        if odom is None:
            self.last_status = "waiting_for_odom"
            return False
        frame_id = self.map_frame_override or odom.header.frame_id or "odom"
        width = max(int(round(self.map_size_m / self.resolution)), 1)
        height = width
        center_column = width // 2
        center_row = height // 2
        if self.map_center is None:
            x = float(odom.pose.pose.position.x)
            y = float(odom.pose.pose.position.y)
        else:
            x, y = self.map_center
        geometry = VisualMapGeometry(
            frame_id,
            self.resolution,
            width,
            height,
            x - (center_column + 0.5) * self.resolution,
            y - (center_row + 0.5) * self.resolution,
        )
        initialized = False
        with self.lock:
            if self.geometry is None:
                self.geometry = geometry
                self.visual_map = np.zeros((height, width), dtype=np.float32)
                initialized = True
                self.rospy.loginfo(
                    "Visual map initialized independently: frame=%s center=(%.3f, %.3f) "
                    "size=%dx%d resolution=%.3f",
                    geometry.frame_id,
                    x,
                    y,
                    width,
                    height,
                    geometry.resolution,
                )
        if initialized:
            self.publish_map_snapshot(geometry, self.rospy.Time.now())
        return True

    def reset_callback(self, _request):
        from std_srvs.srv import EmptyResponse

        with self.lock:
            geometry = self.geometry
            if self.geometry is not None:
                self.visual_map = np.zeros(
                    (self.geometry.height, self.geometry.width), dtype=np.float32
                )
            self.latest_detection_count = 0
            self.latest_raw_detection_count = 0
            self.latest_depth_rejected_count = 0
            self.latest_out_of_map_count = 0
            self.latest_map_updated_count = 0
        if geometry is not None:
            self.publish_map_snapshot(geometry, self.rospy.Time.now())
        self.rospy.loginfo("Reset YOLOE global visual map")
        return EmptyResponse()

    def rgb_input_callback(self, _msg):
        with self.lock:
            self.rgb_messages_received += 1

    def depth_input_callback(self, msg):
        with self.lock:
            self.depth_messages_received += 1
            self.latest_depth_msg = msg

    def rgb_latest_callback(self, rgb_msg):
        with self.lock:
            self.rgb_messages_received += 1
            depth_msg = self.latest_depth_msg
        if depth_msg is None:
            self.last_status = "waiting_for_depth_image"
            return
        rgb_stamp = rgb_msg.header.stamp.to_sec()
        depth_stamp = depth_msg.header.stamp.to_sec()
        dt_sec = abs(rgb_stamp - depth_stamp) if rgb_stamp > 0.0 and depth_stamp > 0.0 else 0.0
        with self.lock:
            self.last_rgb_depth_dt_sec = dt_sec
        if self.max_rgb_depth_age_sec >= 0.0 and dt_sec > self.max_rgb_depth_age_sec:
            self.last_status = "waiting_for_recent_depth_image"
            self.rospy.logwarn_throttle(
                5.0,
                "Latest depth image is %.3fs away from RGB stamp; increase "
                "~max_rgb_depth_age_sec or use time-aligned camera topics.",
                dt_sec,
            )
            return
        self.rgb_depth_callback(rgb_msg, depth_msg)

    def rgb_depth_callback(self, rgb_msg, depth_msg):
        if self.stop_event.is_set() or self.rospy.is_shutdown():
            return
        with self.lock:
            self.frames_received += 1
            rgb_stamp = rgb_msg.header.stamp.to_sec()
            depth_stamp = depth_msg.header.stamp.to_sec()
            if rgb_stamp > 0.0 and depth_stamp > 0.0:
                self.last_rgb_depth_dt_sec = abs(rgb_stamp - depth_stamp)
        task = (rgb_msg, depth_msg)
        try:
            self.frame_queue.put_nowait(task)
            return
        except Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except Empty:
                pass
        try:
            self.frame_queue.put_nowait(task)
            self.rospy.logwarn_throttle(
                5.0, "YOLOE inference is slower than RGB-D input; dropped an older frame"
            )
        except Full:
            pass

    def inference_loop(self):
        while not self.stop_event.is_set():
            try:
                task = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            if task is None:
                self.frame_queue.task_done()
                return
            try:
                self.process_rgb_depth(*task)
            except Exception as exc:
                self.last_status = "processing_failed"
                self.rospy.logerr_throttle(5.0, "YOLOE RGB-D processing failed: %s", exc)
            finally:
                self.frame_queue.task_done()

    def process_rgb_depth(self, rgb_msg, depth_msg):
        with self.lock:
            camera_info = self.latest_camera_info
        if camera_info is None:
            self.last_status = "waiting_for_camera_info"
            return
        if not self.initialize_fallback_geometry():
            return
        if not self.load_model():
            return
        try:
            rgb = ros_rgb_to_bgr(rgb_msg)
            depth = ros_image_to_numpy(depth_msg)
            depth_m = self.convert_depth_to_meters(depth, depth_msg.encoding)
        except Exception as exc:
            self.last_status = "image_decode_failed"
            self.rospy.logwarn_throttle(5.0, "Cannot convert RGB-D image: %s", exc)
            return
        camera_info_scale = (1.0, 1.0)
        if depth_m.shape[:2] != rgb.shape[:2]:
            if not self.resize_rgb_to_depth:
                self.last_status = "rgb_depth_not_aligned"
                self.rospy.logwarn_throttle(
                    5.0,
                    "RGB shape %s and depth shape %s differ and ~resize_rgb_to_depth is false.",
                    rgb.shape[:2],
                    depth_m.shape[:2],
                )
                return
            source_shape = rgb.shape[:2]
            target_shape = depth_m.shape[:2]
            camera_info_scale = (
                float(target_shape[1]) / float(source_shape[1]),
                float(target_shape[0]) / float(source_shape[0]),
            )
            interpolation = (
                cv2.INTER_AREA
                if target_shape[0] < source_shape[0] or target_shape[1] < source_shape[1]
                else cv2.INTER_LINEAR
            )
            rgb = cv2.resize(
                rgb,
                (target_shape[1], target_shape[0]),
                interpolation=interpolation,
            )
            self.rospy.logwarn_throttle(
                5.0,
                "Resized RGB from %s to depth shape %s while preserving original depth values. "
                "For accurate world positions, provide depth already registered to RGB.",
                source_shape,
                target_shape,
            )

        try:
            device = self.device
            if isinstance(device, str) and not device.strip():
                device = None
            results = self.model.predict(
                source=rgb,
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                device=device,
                verbose=False,
                retina_masks=True,
            )
            result = results[0]
        except Exception as exc:
            self.last_status = "inference_failed"
            self.rospy.logerr_throttle(5.0, "YOLOE inference failed: %s", exc)
            return

        self.frames_inferred += 1
        self.latest_raw_detection_count = (
            int(len(result.boxes)) if result.boxes is not None else 0
        )
        result = self.filter_result_by_classes(result)
        self.publish_annotated_image(result, rgb_msg.header)
        self.latest_depth_rejected_count = 0
        self.latest_out_of_map_count = 0
        self.latest_map_updated_count = 0

        with self.lock:
            geometry = self.geometry
            odom = self.latest_odom
        if self.projection_pose_source == "odom":
            if odom is None:
                self.last_status = "waiting_for_odom"
                return
            odom_frame = odom.header.frame_id or geometry.frame_id
            if odom_frame != geometry.frame_id:
                self.last_status = "odom_map_frame_mismatch"
                self.rospy.logwarn_throttle(
                    5.0,
                    "Cannot use odom visual projection: odom frame '%s' differs from "
                    "visual map frame '%s'.",
                    odom_frame,
                    geometry.frame_id,
                )
                return
            point_transform = lambda points: transform_points_with_pose(
                optical_points_to_robot(points), odom.pose.pose
            )
        else:
            camera_frame = (
                self.camera_frame_override
                or camera_info.header.frame_id
                or rgb_msg.header.frame_id
                or depth_msg.header.frame_id
            )
            try:
                transform = self.tf_buffer.lookup_transform(
                    geometry.frame_id,
                    camera_frame,
                    rgb_msg.header.stamp,
                    self.rospy.Duration(self.tf_timeout),
                )
            except Exception as exc:
                self.last_status = "waiting_for_camera_to_map_tf"
                self.rospy.logwarn_throttle(
                    5.0,
                    "YOLOE annotated images are available, but no TF from camera frame "
                    "'%s' to visual map frame '%s': %s",
                    camera_frame,
                    geometry.frame_id,
                    exc,
                )
                return
            point_transform = lambda points: transform_points(points, transform)

        detections = self.extract_detections(
            result, rgb.shape[:2], depth_m, camera_info, point_transform, camera_info_scale
        )
        stamp = rgb_msg.header.stamp if rgb_msg.header.stamp.to_sec() > 0.0 else self.rospy.Time.now()
        updated_count, out_of_map_count = self.update_and_publish_map(
            detections, geometry, stamp
        )
        self.frames_processed += 1
        self.latest_detection_count = len(detections)
        self.latest_map_updated_count = updated_count
        self.latest_out_of_map_count = out_of_map_count
        self.last_status = "running"

    @staticmethod
    def result_class_label(names, class_id):
        class_id = int(class_id)
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        try:
            return str(names[class_id])
        except Exception:
            return str(class_id)

    def class_allowed(self, label, class_id):
        if not self.class_filter_enabled:
            return True
        normalized = normalize_class_name(label)
        class_id_text = str(int(class_id))
        return normalized in self.class_filter or class_id_text in self.class_filter

    def filter_result_by_classes(self, result):
        if not self.class_filter_enabled or result.boxes is None:
            return result
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
        keep_indices = [
            index
            for index, class_id in enumerate(class_ids)
            if self.class_allowed(self.result_class_label(result.names, class_id), class_id)
        ]
        if len(keep_indices) == len(class_ids):
            return result
        try:
            return result[keep_indices]
        except Exception as exc:
            self.rospy.logwarn_throttle(
                5.0,
                "Cannot slice YOLO result for class filter %s: %s. "
                "Annotated image may still show all detections, but map and markers are filtered.",
                sorted(self.class_filter),
                exc,
            )
            return result

    def shutdown(self):
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.rgb_sub.unregister()
        self.depth_sub.unregister()
        while True:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.task_done()
            except Empty:
                break
        try:
            self.frame_queue.put_nowait(None)
        except Full:
            pass
        if self.worker.is_alive() and threading.current_thread() is not self.worker:
            self.worker.join(timeout=self.shutdown_timeout)
        if self.worker.is_alive():
            self.rospy.logwarn(
                "YOLOE worker did not stop within %.2fs; leaving daemon worker during shutdown",
                self.shutdown_timeout,
            )
        else:
            self.rospy.loginfo("YOLOE visual map worker stopped")

    def convert_depth_to_meters(self, depth, encoding):
        depth_array = np.asarray(depth)
        if self.depth_scale > 0.0:
            scale = self.depth_scale
        elif (
            "16U" in str(encoding).upper()
            or "16S" in str(encoding).upper()
            or depth_array.dtype in (np.uint16, np.int16)
        ):
            scale = 0.001
        else:
            scale = 1.0
        return depth_array.astype(np.float32) * float(scale)

    def extract_detections(
        self, result, image_shape, depth_m, camera_info, point_transform, camera_info_scale=(1.0, 1.0)
    ):
        if result.boxes is None:
            return []
        masks = (
            result.masks.data.detach().cpu().numpy()
            if result.masks is not None
            else None
        )
        boxes = result.boxes
        xyxy_boxes = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy().astype(np.int32)
        names = result.names
        height, width = image_shape
        scale_x, scale_y = camera_info_scale
        fx = float(camera_info.K[0]) * scale_x
        fy = float(camera_info.K[4]) * scale_y
        cx = (float(camera_info.K[2]) + 0.5) * scale_x - 0.5
        cy = (float(camera_info.K[5]) + 0.5) * scale_y - 0.5
        if fx <= 0.0 or fy <= 0.0:
            self.rospy.logwarn_throttle(5.0, "CameraInfo has invalid focal length")
            self.latest_depth_rejected_count = int(len(confidences))
            return []

        detections = []
        depth_rejected_count = 0
        for index, (confidence, class_id) in enumerate(zip(confidences, class_ids)):
            if float(confidence) < self.confidence_threshold:
                continue
            label = self.result_class_label(names, class_id)
            if not self.class_allowed(label, class_id):
                continue
            if masks is not None:
                mask = masks[index]
                if mask.shape != (height, width):
                    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                active = mask > 0.5
            else:
                active = self.detection_box_region(xyxy_boxes[index], height, width)
            center_point, valid_depth_points = self.robust_center_depth_point(
                active, depth_m, fx, fy, cx, cy
            )
            if center_point is None:
                depth_rejected_count += 1
                continue
            footprint_points = (
                self.median_depth_footprint_points(active, center_point, fx, fy, cx, cy)
                if self.project_mask_footprint
                else center_point[None, :]
            )
            world_points = point_transform(footprint_points)
            center = point_transform(center_point[None, :])[0]
            detections.append(
                {
                    "class_id": int(class_id),
                    "label": str(label),
                    "confidence": float(confidence),
                    "points_xy": world_points[:, :2],
                    "center": center,
                    "valid_depth_points": int(valid_depth_points),
                    "median_depth_m": float(center_point[2]),
                }
            )
        self.latest_depth_rejected_count = depth_rejected_count
        if depth_rejected_count:
            self.rospy.logwarn_throttle(
                5.0,
                "%d visual object(s) were detected in RGB but skipped because no valid "
                "depth was available in the object region.",
                depth_rejected_count,
            )
        return detections

    def robust_center_depth_point(self, active, depth_m, fx, fy, cx, cy):
        """Project one stable object point from central-mask median depth."""
        rows, columns = np.nonzero(active)
        if rows.size == 0:
            return None, 0
        center_row = float(np.median(rows))
        center_column = float(np.median(columns))
        row_half = max(0.5 * (rows.max() - rows.min() + 1) * self.center_depth_region_scale, 1.0)
        col_half = max(0.5 * (columns.max() - columns.min() + 1) * self.center_depth_region_scale, 1.0)
        central = (
            active
            & (np.abs(np.arange(active.shape[0])[:, None] - center_row) <= row_half)
            & (np.abs(np.arange(active.shape[1])[None, :] - center_column) <= col_half)
        )
        central_rows, central_columns = np.nonzero(central)
        central_depth = depth_m[central_rows, central_columns]
        valid = (
            np.isfinite(central_depth)
            & (central_depth >= self.min_depth_m)
            & (central_depth <= self.max_depth_m)
        )
        valid_depth = central_depth[valid]
        if valid_depth.size < self.min_mask_depth_points:
            all_depth = depth_m[rows, columns]
            valid = (
                np.isfinite(all_depth)
                & (all_depth >= self.min_depth_m)
                & (all_depth <= self.max_depth_m)
            )
            valid_depth = all_depth[valid]
        if valid_depth.size < self.min_mask_depth_points:
            return None, int(valid_depth.size)
        z = float(np.median(valid_depth))
        point = np.asarray(
            [
                (center_column - cx) * z / fx,
                (center_row - cy) * z / fy,
                z,
            ],
            dtype=np.float32,
        )
        return point, int(valid_depth.size)

    def median_depth_footprint_points(self, active, center_point, fx, fy, cx, cy):
        """Project the segmentation footprint using one robust depth for the object."""
        stride = self.footprint_stride_px
        sampled_rows, sampled_columns = np.nonzero(active[::stride, ::stride])
        if sampled_rows.size == 0:
            return center_point[None, :]
        sampled_rows = sampled_rows * stride
        sampled_columns = sampled_columns * stride
        if sampled_rows.size > self.max_footprint_points:
            selected = np.linspace(
                0, sampled_rows.size - 1, self.max_footprint_points, dtype=np.int32
            )
            sampled_rows = sampled_rows[selected]
            sampled_columns = sampled_columns[selected]
        z = float(center_point[2])
        footprint = np.column_stack(
            (
                (sampled_columns.astype(np.float32) - cx) * z / fx,
                (sampled_rows.astype(np.float32) - cy) * z / fy,
                np.full(sampled_rows.shape, z, dtype=np.float32),
            )
        )
        return np.vstack((center_point[None, :], footprint)).astype(np.float32)

    @staticmethod
    def detection_box_region(box, height, width):
        """Build a conservative depth region when an engine returns boxes without masks."""
        x1, y1, x2, y2 = [float(value) for value in box]
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        left = int(np.clip(x1 + 0.25 * box_width, 0, width - 1))
        right = int(np.clip(x2 - 0.25 * box_width, left + 1, width))
        top = int(np.clip(y1 + 0.35 * box_height, 0, height - 1))
        bottom = int(np.clip(y2 - 0.10 * box_height, top + 1, height))
        region = np.zeros((height, width), dtype=bool)
        region[top:bottom, left:right] = True
        return region

    def update_and_publish_map(self, detections, geometry, stamp):
        updated_count = 0
        out_of_map_count = 0
        with self.lock:
            occupied = self.visual_map.copy()
            inflation_cells = int(math.ceil(self.footprint_inflation_m / geometry.resolution))
            for detection in detections:
                rows, columns = geometry.points_to_cells(detection["points_xy"])
                if not rows.size:
                    out_of_map_count += 1
                    continue
                updated_count += 1
                if inflation_cells <= 0:
                    occupied[rows, columns] = self.occupied_value
                else:
                    for row_offset in range(-inflation_cells, inflation_cells + 1):
                        for column_offset in range(-inflation_cells, inflation_cells + 1):
                            if (
                                row_offset * row_offset + column_offset * column_offset
                                > inflation_cells * inflation_cells
                            ):
                                continue
                            expanded_rows = rows + row_offset
                            expanded_columns = columns + column_offset
                            valid = (
                                (expanded_rows >= 0)
                                & (expanded_rows < geometry.height)
                                & (expanded_columns >= 0)
                                & (expanded_columns < geometry.width)
                            )
                            occupied[expanded_rows[valid], expanded_columns[valid]] = (
                                self.occupied_value
                            )
            self.visual_map = np.clip(occupied, 0.0, 1.0)
            map_copy = self.visual_map.copy()
        if out_of_map_count:
            self.rospy.logwarn_throttle(
                5.0,
                "%d projected visual object(s) are outside the visual map. "
                "Increase ~map_size_m or change ~map_center_x/~map_center_y.",
                out_of_map_count,
            )
        self.map_pub.publish(self.make_grid(map_copy, geometry, stamp))
        self.marker_pub.publish(self.make_markers(map_copy, detections, geometry, stamp))
        self.publish_detection_json(detections, geometry, stamp)
        return updated_count, out_of_map_count

    def publish_map_snapshot(self, geometry, stamp):
        """Publish an initialized visual map before TF-dependent observations arrive."""
        with self.lock:
            map_copy = self.visual_map.copy()
        self.map_pub.publish(self.make_grid(map_copy, geometry, stamp))
        self.marker_pub.publish(self.make_markers(map_copy, [], geometry, stamp))
        self.publish_detection_json([], geometry, stamp)

    def make_grid(self, probability, geometry, stamp):
        from nav_msgs.msg import OccupancyGrid

        grid = OccupancyGrid()
        grid.header.frame_id = geometry.frame_id
        grid.header.stamp = stamp
        grid.info.resolution = geometry.resolution
        grid.info.width = geometry.width
        grid.info.height = geometry.height
        grid.info.origin.position.x = geometry.origin_x
        grid.info.origin.position.y = geometry.origin_y
        grid.info.origin.orientation.w = 1.0
        grid.data = np.rint(100.0 * probability).astype(np.int8).reshape(-1).tolist()
        return grid

    def make_markers(self, probability, detections, geometry, stamp):
        from geometry_msgs.msg import Point
        from std_msgs.msg import ColorRGBA
        from visualization_msgs.msg import Marker, MarkerArray

        with self.lock:
            odom = self.latest_odom
            robot_path = list(self.robot_path)
            previous_object_marker_count = self.last_object_marker_count
            self.last_object_marker_count = len(detections)

        heatmap = Marker()
        heatmap.header.frame_id = geometry.frame_id
        heatmap.header.stamp = stamp
        heatmap.ns = "visual_map_heatmap"
        heatmap.id = 0
        heatmap.type = Marker.CUBE_LIST
        heatmap.action = Marker.ADD
        heatmap.pose.orientation.w = 1.0
        heatmap.scale.x = self.marker_cell_size_m
        heatmap.scale.y = self.marker_cell_size_m
        heatmap.scale.z = max(geometry.resolution * 0.10, 0.01)
        heatmap.color.a = 1.0

        selected = np.flatnonzero(probability.reshape(-1) >= self.publish_marker_threshold)
        if selected.size > self.marker_max_cells:
            values = probability.reshape(-1)[selected]
            top = np.argpartition(values, -self.marker_max_cells)[-self.marker_max_cells:]
            selected = selected[top]
        for flat_index in selected.tolist():
            row, column = np.unravel_index(flat_index, probability.shape)
            value = float(probability[row, column])
            x, y = geometry.cell_center(row, column)
            heatmap.points.append(Point(x=x, y=y, z=self.marker_height))
            heatmap.colors.append(
                ColorRGBA(r=0.10, g=min(1.0, 0.30 + value), b=1.0 - 0.75 * value, a=0.20 + 0.65 * value)
            )

        markers = []
        if heatmap.points:
            markers.append(heatmap)
        else:
            heatmap.action = Marker.DELETE
            markers.append(heatmap)

        odom_frame = odom.header.frame_id if odom is not None else ""
        if odom is not None and (not odom_frame or odom_frame == geometry.frame_id):
            robot = Marker()
            robot.header = heatmap.header
            robot.ns = "visual_map_robot"
            robot.id = 0
            robot.type = Marker.ARROW
            robot.action = Marker.ADD
            robot.pose.position = Point(
                x=float(odom.pose.pose.position.x),
                y=float(odom.pose.pose.position.y),
                z=self.robot_marker_height,
            )
            robot.pose.orientation = odom.pose.pose.orientation
            robot.scale.x = 0.60
            robot.scale.y = 0.10
            robot.scale.z = 0.10
            robot.color = ColorRGBA(r=0.05, g=0.80, b=1.00, a=0.95)
            markers.append(robot)

            path_points = [
                Point(x=x, y=y, z=self.robot_marker_height * 0.5)
                for x, y, frame_id in robot_path
                if not frame_id or frame_id == geometry.frame_id
            ]
            if len(path_points) >= 2:
                trajectory = Marker()
                trajectory.header = heatmap.header
                trajectory.ns = "visual_map_robot"
                trajectory.id = 1
                trajectory.type = Marker.LINE_STRIP
                trajectory.action = Marker.ADD
                trajectory.pose.orientation.w = 1.0
                trajectory.scale.x = 0.045
                trajectory.color = ColorRGBA(r=0.05, g=0.80, b=1.00, a=0.80)
                trajectory.points = path_points
                markers.append(trajectory)

        for index, detection in enumerate(detections):
            red, green, blue = color_for_class(detection["class_id"])
            center = detection["center"]
            object_marker = Marker()
            object_marker.header = heatmap.header
            object_marker.ns = "visual_objects"
            object_marker.id = 2 * index
            object_marker.type = Marker.SPHERE
            object_marker.action = Marker.ADD
            object_marker.pose.position = Point(
                x=float(center[0]), y=float(center[1]), z=self.object_marker_height
            )
            object_marker.pose.orientation.w = 1.0
            object_marker.scale.x = object_marker.scale.y = object_marker.scale.z = 0.22
            object_marker.color = ColorRGBA(r=red, g=green, b=blue, a=0.95)
            markers.append(object_marker)

            label = Marker()
            label.header = heatmap.header
            label.ns = "visual_objects"
            label.id = 2 * index + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position = Point(
                x=float(center[0]), y=float(center[1]), z=self.object_marker_height + 0.30
            )
            label.pose.orientation.w = 1.0
            label.scale.z = 0.18
            label.color = ColorRGBA(r=red, g=green, b=blue, a=1.0)
            label.text = "%s %.2f\n(%.2f, %.2f)" % (
                detection["label"],
                detection["confidence"],
                center[0],
                center[1],
            )
            markers.append(label)
        for index in range(len(detections), previous_object_marker_count):
            for marker_id in (2 * index, 2 * index + 1):
                deletion = Marker()
                deletion.header = heatmap.header
                deletion.ns = "visual_objects"
                deletion.id = marker_id
                deletion.action = Marker.DELETE
                markers.append(deletion)
        return MarkerArray(markers=markers)

    def publish_annotated_image(self, result, header):
        try:
            from sensor_msgs.msg import Image

            image = result.plot()
            message = Image()
            message.header = header
            message.height = int(image.shape[0])
            message.width = int(image.shape[1])
            message.encoding = "bgr8"
            message.is_bigendian = 0
            message.step = int(image.shape[1] * image.shape[2])
            message.data = np.ascontiguousarray(image, dtype=np.uint8).tobytes()
            self.annotated_pub.publish(message)
        except Exception as exc:
            self.rospy.logwarn_throttle(5.0, "Cannot publish YOLOE annotated image: %s", exc)

    def publish_detection_json(self, detections, geometry, stamp):
        from std_msgs.msg import String

        payload = {
            "stamp": stamp.to_sec(),
            "frame_id": geometry.frame_id,
            "count": len(detections),
            "detections": [
                {
                    "label": detection["label"],
                    "class_id": detection["class_id"],
                    "confidence": detection["confidence"],
                    "x": float(detection["center"][0]),
                    "y": float(detection["center"][1]),
                    "z": float(detection["center"][2]),
                    "valid_depth_points": detection["valid_depth_points"],
                }
                for detection in detections
            ],
        }
        self.detections_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))

    def status_callback(self, _event):
        from std_msgs.msg import String

        with self.lock:
            geometry = self.geometry
            has_camera_info = self.latest_camera_info is not None
            has_odom = self.latest_odom is not None
            geometry_from_audio_map = self.geometry_from_audio_map
            occupied_cell_count = (
                int(np.count_nonzero(self.visual_map >= self.publish_marker_threshold))
                if self.visual_map is not None
                else 0
            )
        payload = {
            "status": self.last_status,
            "model": self.model_path,
            "model_loaded": self.model is not None,
            "model_error": self.model_error,
            "classes": self.classes,
            "class_filter_enabled": self.class_filter_enabled,
            "class_filter": sorted(self.class_filter),
            "confidence_threshold": self.confidence_threshold,
            "projection_pose_source": self.projection_pose_source,
            "broadcast_odom_tf": self.broadcast_odom_tf,
            "sensor_frame_id": self.sensor_frame_id,
            "project_mask_footprint": self.project_mask_footprint,
            "footprint_inflation_m": self.footprint_inflation_m,
            "image_pairing_mode": self.image_pairing_mode,
            "rgb_messages_received": int(self.rgb_messages_received),
            "depth_messages_received": int(self.depth_messages_received),
            "frames_received": int(self.frames_received),
            "frames_inferred": int(self.frames_inferred),
            "frames_processed": int(self.frames_processed),
            "latest_raw_detection_count": int(self.latest_raw_detection_count),
            "latest_detection_count": int(self.latest_detection_count),
            "latest_depth_rejected_count": int(self.latest_depth_rejected_count),
            "latest_out_of_map_count": int(self.latest_out_of_map_count),
            "latest_map_updated_count": int(self.latest_map_updated_count),
            "occupied_cell_count": occupied_cell_count,
            "has_camera_info": has_camera_info,
            "has_odom": has_odom,
            "has_geometry": geometry is not None,
            "geometry_from_audio_map": geometry_from_audio_map,
            "frame_id": geometry.frame_id if geometry is not None else "",
            "last_rgb_depth_dt_sec": self.last_rgb_depth_dt_sec,
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=True)))


def main():
    import rospy

    rospy.init_node("yoloe_visual_map")
    YOLOEVisualMapNode()
    rospy.spin()


if __name__ == "__main__":
    main()
