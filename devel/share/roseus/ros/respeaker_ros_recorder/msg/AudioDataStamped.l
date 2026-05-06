;; Auto-generated. Do not edit!


(when (boundp 'respeaker_ros_recorder::AudioDataStamped)
  (if (not (find-package "RESPEAKER_ROS_RECORDER"))
    (make-package "RESPEAKER_ROS_RECORDER"))
  (shadow 'AudioDataStamped (find-package "RESPEAKER_ROS_RECORDER")))
(unless (find-package "RESPEAKER_ROS_RECORDER::AUDIODATASTAMPED")
  (make-package "RESPEAKER_ROS_RECORDER::AUDIODATASTAMPED"))

(in-package "ROS")
;;//! \htmlinclude AudioDataStamped.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass respeaker_ros_recorder::AudioDataStamped
  :super ros::object
  :slots (_header _sample_rate _channels _frames _data ))

(defmethod respeaker_ros_recorder::AudioDataStamped
  (:init
   (&key
    ((:header __header) (instance std_msgs::Header :init))
    ((:sample_rate __sample_rate) 0)
    ((:channels __channels) 0)
    ((:frames __frames) 0)
    ((:data __data) (make-array 0 :initial-element 0 :element-type :integer))
    )
   (send-super :init)
   (setq _header __header)
   (setq _sample_rate (round __sample_rate))
   (setq _channels (round __channels))
   (setq _frames (round __frames))
   (setq _data __data)
   self)
  (:header
   (&rest __header)
   (if (keywordp (car __header))
       (send* _header __header)
     (progn
       (if __header (setq _header (car __header)))
       _header)))
  (:sample_rate
   (&optional __sample_rate)
   (if __sample_rate (setq _sample_rate __sample_rate)) _sample_rate)
  (:channels
   (&optional __channels)
   (if __channels (setq _channels __channels)) _channels)
  (:frames
   (&optional __frames)
   (if __frames (setq _frames __frames)) _frames)
  (:data
   (&optional __data)
   (if __data (setq _data __data)) _data)
  (:serialization-length
   ()
   (+
    ;; std_msgs/Header _header
    (send _header :serialization-length)
    ;; uint32 _sample_rate
    4
    ;; uint32 _channels
    4
    ;; uint32 _frames
    4
    ;; int16[] _data
    (* 2    (length _data)) 4
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Header _header
       (send _header :serialize s)
     ;; uint32 _sample_rate
       (write-long _sample_rate s)
     ;; uint32 _channels
       (write-long _channels s)
     ;; uint32 _frames
       (write-long _frames s)
     ;; int16[] _data
     (write-long (length _data) s)
     (dotimes (i (length _data))
       (write-word (elt _data i) s)
       )
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Header _header
     (send _header :deserialize buf ptr-) (incf ptr- (send _header :serialization-length))
   ;; uint32 _sample_rate
     (setq _sample_rate (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _channels
     (setq _channels (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; uint32 _frames
     (setq _frames (sys::peek buf ptr- :integer)) (incf ptr- 4)
   ;; int16[] _data
   (let (n)
     (setq n (sys::peek buf ptr- :integer)) (incf ptr- 4)
     (setq _data (instantiate integer-vector n))
     (dotimes (i n)
     (setf (elt _data i) (sys::peek buf ptr- :short)) (incf ptr- 2)
     ))
   ;;
   self)
  )

(setf (get respeaker_ros_recorder::AudioDataStamped :md5sum-) "54015d5fe0aee8071806c47e9f87c566")
(setf (get respeaker_ros_recorder::AudioDataStamped :datatype-) "respeaker_ros_recorder/AudioDataStamped")
(setf (get respeaker_ros_recorder::AudioDataStamped :definition-)
      "std_msgs/Header header
uint32 sample_rate
uint32 channels
uint32 frames
int16[] data

================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :respeaker_ros_recorder/AudioDataStamped "54015d5fe0aee8071806c47e9f87c566")


