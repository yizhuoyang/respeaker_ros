
(cl:in-package :asdf)

(defsystem "respeaker_ros_recorder-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :std_msgs-msg
)
  :components ((:file "_package")
    (:file "AudioDataStamped" :depends-on ("_package_AudioDataStamped"))
    (:file "_package_AudioDataStamped" :depends-on ("_package"))
  ))