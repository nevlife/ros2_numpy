import os
import time

import numpy as np
import pytest
import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from ros2_numpy.type_adapter import POINT_CLOUD_ADAPTER, PointCloud
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

DTYPE = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'u2')]


def cloud(shape):
    points = np.zeros(shape, dtype=DTYPE)
    points['x'] = np.arange(points.size).reshape(shape)
    points['intensity'] = 7
    return points


def test_conversions_keep_points_and_header():
    header = Header(frame_id='lidar')
    header.stamp.sec = 3
    for shape in [(5,), (2, 3)]:
        message = POINT_CLOUD_ADAPTER.to_ros(PointCloud(cloud(shape), header))
        assert message.header.frame_id == 'lidar' and message.header.stamp.sec == 3
        assert (message.height, message.width) == ((1,) + shape)[-2:]
        value = POINT_CLOUD_ADAPTER.from_ros(message)
        assert value.points.shape == shape and value.header.frame_id == 'lidar'
        np.testing.assert_array_equal(value.points, cloud(shape))


@pytest.mark.parametrize('broker', [False, True], ids=['rmw', 'broker'])
def test_node_delivers_point_clouds(broker, monkeypatch):
    monkeypatch.setenv('RCL_INTRA_PROCESS', '1' if broker else '0')
    broker = broker and os.environ.get('RMW_IMPLEMENTATION') == 'rmw_zenoh_cpp'
    context = Context()
    rclpy.init(context=context)
    node = rclpy.create_node('point_cloud_adapter', context=context)
    executor = SingleThreadedExecutor(context=context)
    received, messages = [], []
    try:
        publisher = node.create_publisher(PointCloud, 'points', 10)
        node.create_subscription(PointCloud, 'points', received.append, 10)
        node.create_subscription(PointCloud2, 'points', messages.append, 10)
        executor.add_node(node)
        deadline = time.monotonic() + 5
        while publisher.get_subscription_count() < 2:
            assert time.monotonic() < deadline
            executor.spin_once(timeout_sec=0.01)
        sent = PointCloud(cloud((100,)), Header(frame_id='lidar'))
        publisher.publish(sent)
        while not received or not messages:
            assert time.monotonic() < deadline
            executor.spin_once(timeout_sec=0.01)
        assert (received[0] is sent) == broker
        np.testing.assert_array_equal(received[0].points, sent.points)
        assert received[0].header.frame_id == 'lidar'
        assert messages[0].width == 100 and messages[0].point_step == 14
    finally:
        executor.shutdown()
        node.destroy_node()
        context.shutdown()
