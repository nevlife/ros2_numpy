"""
An rclpy type adapter between NumPy point arrays and sensor_msgs/PointCloud2.

Importing this module registers ``POINT_CLOUD_ADAPTER`` for :class:`PointCloud`, so
``node.create_publisher(PointCloud, ...)`` publishes clouds and
``node.create_subscription(PointCloud, ...)`` delivers them, like the C++
``pcl::PointCloud`` adapter of pcl_conversions. In one process with the rcl broker, a
subscription receives the published :class:`PointCloud` itself.
"""

from dataclasses import dataclass, field

import numpy as np
from rclpy.type_adapter import register_type_adapter, TypeAdapter
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from .point_cloud2 import array_to_pointcloud2, pointcloud2_to_array


@dataclass
class PointCloud:
    """
    Points as a NumPy structured array with the header of their message.

    :param points: One field per point field, shape ``(width,)`` or ``(height, width)``.
    :param header: The header of the message.
    """

    points: np.ndarray
    header: Header = field(default_factory=Header)


def point_cloud_to_msg(value):
    """Convert a :class:`PointCloud` to a sensor_msgs/PointCloud2."""
    return array_to_pointcloud2(value.points, value.header.stamp, value.header.frame_id)


def msg_to_point_cloud(message):
    """
    Wrap a sensor_msgs/PointCloud2 in a :class:`PointCloud` without copying its points.

    ``points`` views the message data. With ``RCLPY_MESSAGE_ARRAYS=numpy`` it views the
    received message without any copy and is read-only.
    """
    return PointCloud(pointcloud2_to_array(message), message.header)


POINT_CLOUD_ADAPTER = TypeAdapter(PointCloud, PointCloud2, point_cloud_to_msg, msg_to_point_cloud)
register_type_adapter(POINT_CLOUD_ADAPTER)
