"""HSV colour detection on the overhead camera.

Subscribes  /overhead_camera/image_raw   sensor_msgs/Image
Publishes   /detected_objects            scara_msgs/DetectedObjectArray
            /detected_markers            visualization_msgs/MarkerArray  (RViz)
            /vision/debug_image          sensor_msgs/Image  (masks overlaid)

Red wraps around the hue axis in OpenCV's 0..179 scale, so it needs two
ranges -- that is the one part of HSV thresholding people usually get wrong.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from scara_msgs.msg import DetectedObject, DetectedObjectArray

from . import scara_kinematics as K

# (low, high) pairs in OpenCV HSV: H 0..179, S 0..255, V 0..255.
HSV_RANGES = {
   "red":   [((0, 70, 40), (12, 255, 255)),
              ((168, 70, 40), (179, 255, 255))],
    "green": [((40, 90, 60), (85, 255, 255))],
    "blue":  [((95, 120, 60), (130, 255, 255))],
}

MIN_AREA = 60      # px, rejects speckle
MAX_AREA = 6000    # px, rejects the bin discs if they ever pass the filter

# Nominal top-face area (px^2) of an unoccluded cube, from the same pinhole
# geometry pixel_to_world() uses -- confidence is scaled against this actual
# expected size instead of an arbitrary constant.
_height_m = K.CAM_EYE_Z - K.CUBE_SIZE
_extent_m = 2.0 * _height_m * np.tan(np.radians(K.CAM_FOV) / 2.0)
_px_per_m = K.CAM_H / _extent_m
NOMINAL_CUBE_AREA = (K.CUBE_SIZE * _px_per_m) ** 2


class VisionNode(Node):
    def __init__(self):
        super().__init__("scara_vision")
        self.bridge = CvBridge()
        self.pub = self.create_publisher(DetectedObjectArray, "/detected_objects", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/detected_markers", 10)
        self.pub_dbg = self.create_publisher(Image, "/vision/debug_image", 2)
        self.create_subscription(Image, "/overhead_camera/image_raw",
                                 self.on_image, 2)
        self.get_logger().info("vision node up")

    def on_image(self, msg):
        rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        out = DetectedObjectArray()
        out.header = msg.header
        dbg = bgr.copy()

        for color, ranges in HSV_RANGES.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, np.array(lo, np.uint8),
                                    np.array(hi, np.uint8))
            # close pinholes, then drop specks
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if not (MIN_AREA <= area <= MAX_AREA):
                    continue
                m = cv2.moments(c)
                if m["m00"] == 0:
                    continue
                u = m["m10"] / m["m00"]
                v = m["m01"] / m["m00"]
                x, y = K.pixel_to_world(u, v)

                # the bins are flat discs on the ground in the same colours;
                # reject anything sitting on top of a bin centre
                if min(float(np.hypot(x - bx, y - by))
                       for bx, by in K.BIN_XY) < K.BIN_TOL:
                    continue

                d = DetectedObject()
                d.color = color
                d.position.x = x
                d.position.y = y
                d.position.z = K.CUBE_SIZE / 2.0
                d.confidence = float(min(1.0, area / NOMINAL_CUBE_AREA))
                out.objects.append(d)

                cv2.drawContours(dbg, [c], -1, (255, 255, 255), 1)
                cv2.putText(dbg, f"{color} {x:+.2f},{y:+.2f}",
                            (int(u) - 40, int(v) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        self.pub.publish(out)
        self.publish_markers(out)
        dmsg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
        dmsg.header = msg.header
        self.pub_dbg.publish(dmsg)

    def publish_markers(self, det):
        ma = MarkerArray()
        for i, o in enumerate(det.objects):
            mk = Marker()
            mk.header.frame_id = "world"
            mk.header.stamp = det.header.stamp
            mk.ns = o.color
            mk.id = i
            mk.type = Marker.CUBE
            mk.action = Marker.ADD
            mk.pose.position = o.position
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = K.CUBE_SIZE
            r, g, b, a = K.COLOR_RGBA[o.color]
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = r, g, b, 0.8
            ma.markers.append(mk)
        self.pub_mk.publish(ma)


def main():
    rclpy.init()
    node = VisionNode()
    try:
        # rclpy.spin(node)'s blocking wait does not reliably deliver
        # subscribed/published data on this host (WSL2 + rmw_fastrtps_cpp);
        # a manually-timed poll loop does.
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
